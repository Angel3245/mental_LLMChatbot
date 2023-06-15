import torch
import torch.nn as nn
import sys, os, csv
from tqdm import tqdm
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments, get_scheduler, DataCollatorForSeq2Seq, GenerationConfig
from sklearn.model_selection import train_test_split
from shared.prompter import Prompter
from shared import make_dirs

from transformers import BloomTokenizerFast
from petals import DistributedBloomForCausalLM

from shared.prompter import Prompter

from peft import (
    prepare_model_for_int8_training,
    PrefixTuningConfig,
    get_peft_model,
    get_peft_config,
    get_peft_model_state_dict,
    PeftModel,
    PeftType,
    TaskType
)

import logging
logging.basicConfig(
        format="%(asctime)s — %(levelname)s — %(name)s — %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

class BloomBasedChatbot(nn.Module):
  def __init__(
      self,
      model,
      intermediate_size: int = 32,
      num_classes: int = 2,
      adapter_layer_position: int = 6,
      head_layer_position: int = 10
    ):
    super().__init__()
    self.distributed_layers = model.transformer.h

    self.hidden_size = model.config.hidden_size
    self.dtype = model.config.torch_dtype
    self.intermediate_size = intermediate_size
    self.num_classes = num_classes
    self.adapter_layer_position = adapter_layer_position
    self.head_layer_position = head_layer_position
    
    self.word_embeddings = model.transformer.word_embeddings
    self.adapter = nn.Sequential(
        nn.Linear(self.hidden_size, self.intermediate_size),
        nn.Linear(self.intermediate_size, self.hidden_size),
    ).to(self.dtype)
    self.head = nn.Sequential(
        nn.LayerNorm(self.hidden_size),
        nn.Linear(self.hidden_size, self.num_classes),
    ).to(self.dtype)
  
  def forward(self, embeddings):
    before_layers = self.distributed_layers[0:self.adapter_layer_position]
    after_layers = self.distributed_layers[self.adapter_layer_position:self.head_layer_position]
    
    hidden_states = before_layers(embeddings)
    hidden_states = self.adapter(hidden_states)
    hidden_states = after_layers(hidden_states)
    pooled_states = torch.mean(hidden_states, dim=1)
    return self.head(pooled_states)
  
class PetalsTrainer:
    def __init__(self, model_name="bigscience/bloom-petals", cutoff_len = 512):
        self.prompter = Prompter("mentalbot")

        self.tokenizer = BloomTokenizerFast.from_pretrained(model_name)
        self.tokenizer.padding_side = "right" # Allow batched inference

        self.model = DistributedBloomForCausalLM.from_pretrained(model_name, pre_seq_len=16, 
            tuning_mode='ptune')

        # training hyperparams
        self.cutoff_len = cutoff_len

        self.model.to("cuda")

    def tokenize(self, prompt, add_eos_token=True):
        if add_eos_token:
            prompt += self.tokenizer.eos_token

        result = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.cutoff_len,
            padding='max_length',
            return_tensors='pt'
        )
    
        result["labels"] = torch.clone(result["input_ids"])

        result.pop("attention_mask")
    
        return result
 
    def generate_and_tokenize_prompt(self,data_point):
        full_prompt = self.prompter.generate_prompt(data_point["prompt"],None,data_point["completion"])
        tokenized_full_prompt = self.tokenize(full_prompt)
        return tokenized_full_prompt

    def train(self, dataset, output_dir, epochs=1, batch_size=4, lr=2e-4, val_set_size=200):
        # split dataset into separate training and validation sets
        train_val = dataset["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )

        # create prompts from the loaded dataset and tokenize them
        train_dataset = (
            train_val["train"].map(self.generate_and_tokenize_prompt)
        )
        val_dataset = (
            train_val["test"].map(self.generate_and_tokenize_prompt)
        )

        # training hyperparams
        gradient_accumulation_steps = 4
        
        training_args = TrainingArguments(
            output_dir=output_dir,          # output directory
            num_train_epochs=epochs,         # total number of training epochs
            per_device_train_batch_size=batch_size,  # batch size per device during training
            per_device_eval_batch_size=batch_size,   # batch size for evaluation
            learning_rate=lr,               # learning rate
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=10,       # number of warmup steps for learning rate scheduler
            #weight_decay=0.01,              # strength of weight decay
            logging_dir='./logs',            # directory for storing logs
            logging_steps=1,
            save_steps=100,                  # after # steps model is saved
            evaluation_strategy='steps',
            save_strategy="steps",
            optim="adamw_torch",
            eval_steps=10,                  # Number of update steps between two evaluations.
            fp16=True,                       # whether to use floating point 16 for training
            fp16_opt_level="O1",             # see apex AMP optimization level for detail
            save_total_limit=3,
            load_best_model_at_end=True,
        )

        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer, pad_to_multiple_of=8, 
            return_tensors="pt",
            padding=True
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )

        self.model.config.use_cache = False

        if torch.__version__ >= "2" and sys.platform != "win32":
            self.model = torch.compile(self.model)

        resume_from_checkpoint = False
        if os.path.exists(output_dir):
            if len([n for n in os.listdir(output_dir) if n.startswith("checkpoint")]):
                resume_from_checkpoint = True
                
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        # Save the model and tokenizer
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
    
    def generate_response(self, input_text, max_new_tokens=256, temperature=0.1, top_p=0.9, top_k=40, num_beams=1, repetition_penalty=1.1):

        self.model = self.model.eval()

        # Set prompt
        prompt = self.prompter.generate_prompt(input_text)

        input_encodings = self.tokenizer(prompt, return_tensors='pt')
        input_ids = input_encodings['input_ids'].to(self.model.device)

        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            #top_k=top_k,
            #num_beams=num_beams,
            repetition_penalty=repetition_penalty
        )
        
        with torch.inference_mode():
            # Use model.generate() to generate the response
            response = self.model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )

        # Decode the response from the model back into text
        decoded_output = self.tokenizer.decode(response.sequences[0])
        response = self.prompter.get_response(decoded_output)

        return response
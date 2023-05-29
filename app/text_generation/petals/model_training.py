import torch
import torch.nn as nn
import numpy as np
import evaluate
from tqdm import tqdm
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments, get_scheduler, DataCollatorForSeq2Seq, GenerationConfig
from sklearn.model_selection import train_test_split
from rouge_score import rouge_scorer
from train_dataset import PetalsDataset
from shared.prompter import Prompter
from shared import make_dirs

from transformers import BloomTokenizerFast
from petals import DistributedBloomForCausalLM

from shared.prompter import Prompter

from peft import (
    prepare_model_for_int8_training,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    PeftModel,
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
        self.prompter = Prompter("chatbot_simple")

        self.tokenizer = BloomTokenizerFast.from_pretrained(model_name)
        self.tokenizer.pad_token_id = 0
        self.tokenizer.padding_side = "left" # Allow batched inference

        """ INTERMEDIATE_SIZE = 32
        ADAPTER_LAYER_POSITION = 6
        HEAD_LAYER_POSITION = 10
        self.model = BloomBasedClassifier(
            DistributedBloomForCausalLM.from_pretrained(model_name),
            intermediate_size=INTERMEDIATE_SIZE,
            adapter_layer_position=ADAPTER_LAYER_POSITION,
            head_layer_position=HEAD_LAYER_POSITION,
        ) """

        self.model = DistributedBloomForCausalLM.from_pretrained(model_name)
        #self.model = prepare_model_for_int8_training(self.model)

        # training hyperparams
        self.cutoff_len = cutoff_len

        # lora hyperparams
        lora_r = 8
        lora_alpha = 16
        lora_dropout = 0.05
        """ lora_target_modules = [
            "adapter",
        ] """

        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            #target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, config)
        self.model.print_trainable_parameters()

        self.model.to("cuda")

    def tokenize(self, prompt, add_eos_token=True):
        result = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != self.tokenizer.eos_token_id
            and len(result["input_ids"]) < self.cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(self.tokenizer.eos_token_id)
            result["attention_mask"].append(1)
    
        result["labels"] = result["input_ids"].copy()
    
        return result
 
    def generate_and_tokenize_prompt(self,data_point):
        full_prompt = self.prompter.generate_prompt(data_point["prompt"],None,data_point["completion"])
        tokenized_full_prompt = self.tokenize(full_prompt)
        return tokenized_full_prompt

    def train(self, dataset, output_dir, epochs=1, batch_size=4, lr=2e-5, val_set_size=200):
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
        full_batch_size = 128
        gradient_accumulation_steps = full_batch_size // batch_size

        # llm hyperparams
        group_by_length = True  # faster, but produces an odd training loss curve
        
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
            group_by_length=group_by_length,
        )

        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )

        self.model.config.use_cache = False

        old_state_dict = self.model.state_dict
        self.model.state_dict = (
            lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
        ).__get__(self.model, type(self.model))

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

    def evaluation(self, test_inputs, output_path, max_new_tokens=256, temperature=0.1, top_p=0.9, top_k=40, num_beams=4, repetition_penalty=1.1):

        self.model = self.model.eval()

        for input_text in test_inputs:
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

            # Create CSV with evaluation results
            make_dirs(output_path)
            with open(output_path+"/evaluation.csv", 'a', encoding="UTF8") as csv_file:
                writer = csv.writer(csv_file, delimiter=",")
                writer.writerow(["Input","Response"])
                writer.writerow([input_text,response])
    
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
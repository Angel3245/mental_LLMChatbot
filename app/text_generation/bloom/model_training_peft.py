import torch
import os, sys, copy
from transformers import Trainer, TrainingArguments, GenerationConfig, BloomTokenizerFast, BloomForCausalLM, DataCollatorForSeq2Seq
from shared.prompter import Prompter
from ray import tune

from peft import (
    prepare_model_for_int8_training,
    LoraConfig,
    get_peft_model,
    PeftModel,
    PeftConfig
)

import logging
logging.basicConfig(
        format="%(asctime)s — %(levelname)s — %(name)s — %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

class BloomPeftTrainer:
    """ Class for finetuning BLOOM using Parameter Efficient Fine-tuning

        :param model_path: path of the trained model in disk
        :param base_model: name of the base model
        :param cutoff_len: max length of sentences
        :param template: template file to use to create prompts
    """
    def __init__(self, model_path=None, base_model=None, cutoff_len = 512, template = "mentalbot"):
        device_map = "auto"

        self.prompter = Prompter(template)

        if not model_path == None:
            # Load Peft model
            print("Loading Peft model from disk")
            config = PeftConfig.from_pretrained(model_path)

            self.model = BloomForCausalLM.from_pretrained(
                config.base_model_name_or_path,
                load_in_8bit=True,
                return_dict=True,
                torch_dtype=torch.float16,
                device_map={"":0},
            )

            self.model = PeftModel.from_pretrained(self.model, model_path, torch_dtype=torch.float16, device_map={"":0})
            self.tokenizer = BloomTokenizerFast.from_pretrained(config.base_model_name_or_path)
            
        else:
            # Create Peft model from Bloom model
            print("Creating Peft model from Bloom model")
            self.model = BloomForCausalLM.from_pretrained(
                base_model,
                load_in_8bit=True,
                torch_dtype=torch.float16,
                device_map=device_map,
            )

            self.model = prepare_model_for_int8_training(self.model)

            # lora hyperparams
            lora_r = 8
            lora_alpha = 16
            lora_dropout = 0.05

            config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(self.model, config)
            self.tokenizer = BloomTokenizerFast.from_pretrained(base_model)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.tokenizer.padding_side = "left" # Allow batched inference

        # training hyperparams
        self.cutoff_len = cutoff_len

        self.model.print_trainable_parameters()

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
        full_prompt = self.prompter.generate_prompt(data_point["prompt"],data_point["completion"])
        tokenized_full_prompt = self.tokenize(full_prompt)
        return tokenized_full_prompt

    def train(self, dataset, output_dir, epochs=2, batch_size=4, lr=5e-4, val_set_size=200):
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
        
        training_args = TrainingArguments(
            output_dir=output_dir,          # output directory
            num_train_epochs=epochs,         # total number of training epochs
            per_device_train_batch_size=batch_size,  # batch size per device during training
            per_device_eval_batch_size=batch_size,   # batch size for evaluation
            learning_rate=lr,               # learning rate
            gradient_accumulation_steps=4,
            #warmup_steps=10,       # number of warmup steps for learning rate scheduler
            #weight_decay=0.01,              # strength of weight decay
            logging_dir='./logs',            # directory for storing logs
            logging_steps=10,
            save_steps=1000,                  # after # steps model is saved
            evaluation_strategy='steps',
            save_strategy="no",
            optim="adamw_torch",
            eval_steps=30,                  # Number of update steps between two evaluations.
            fp16=True,                       # whether to use floating point 16 for training
            fp16_opt_level="O1",             # see apex AMP optimization level for detail
            report_to="tensorboard"
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

    def hyperparameter_search(self, dataset):   
        # split dataset into separate training and validation sets
        train_val = dataset["train"].train_test_split(
            test_size=200, shuffle=True, seed=42
        )

        # create prompts from the loaded dataset and tokenize them
        train_dataset = (
            train_val["train"].map(self.generate_and_tokenize_prompt)
        )
        val_dataset = (
            train_val["test"].map(self.generate_and_tokenize_prompt)
        )

        # llm hyperparams
        group_by_length = False  # faster, but produces an odd training loss curve
        
        training_args = TrainingArguments(
            #warmup_steps=10,       # number of warmup steps for learning rate scheduler
            #weight_decay=0.01,              # strength of weight decay
            output_dir='./logs',            # directory for storing output
            logging_dir='./logs',            # directory for storing logs
            gradient_accumulation_steps=4,
            logging_steps=1,
            evaluation_strategy='steps',
            optim="adamw_torch",
            eval_steps=1000,                  # Number of update steps between two evaluations.
            fp16=True,                       # whether to use floating point 16 for training
            fp16_opt_level="O1",             # see apex AMP optimization level for detail
            group_by_length=group_by_length,
        )

        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        )
        
        def model_init():
            return copy.deepcopy(self.model)
        
        trainer = Trainer(
            model_init=model_init,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )

        self.model.config.use_cache = False

        if torch.__version__ >= "2" and sys.platform != "win32":
            self.model = torch.compile(self.model)
                
        def my_hp_space(trial):
            return {
                "learning_rate": tune.loguniform(1e-6, 1e-3),
                "num_train_epochs": tune.choice([1, 2, 3]),
                "per_device_train_batch_size": tune.choice([2, 4, 8])
            }
        
        # Default objective is the sum of all metrics
        # when metrics are provided, so we have to maximize it.
        best_run = trainer.hyperparameter_search(
            direction="minimize", 
            backend="ray", 
            n_trials=10, # number of trials
            hp_space=my_hp_space
        )

        print(best_run)

    def generate_response(self, input_text, max_new_tokens=256, temperature=0.9, top_p=0.9, repetition_penalty=1.1):

        self.model = self.model.eval()

        # Set prompt
        prompt = self.prompter.generate_prompt(input_text)

        input_encodings = self.tokenizer(prompt, return_tensors='pt')
        input_ids = input_encodings['input_ids'].to(self.model.device)

        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
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
        decoded_output = self.tokenizer.decode(response.sequences[0][ : -1])

        response = self.prompter.get_response(decoded_output)

        return response

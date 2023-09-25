# Copyright (C) 2023  Jose Ángel Pérez Garrido
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import torch
from transformers import Trainer, TrainingArguments
from rouge_score import rouge_scorer
from transformers import BloomTokenizerFast, BloomForCausalLM, DataCollatorForSeq2Seq
from shared.prompter import Prompter

import logging
logging.basicConfig(
        format="%(asctime)s — %(levelname)s — %(name)s — %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

class BloomTrainer:
    """ Class for finetuning GPT2 using Full Fine-tuning

        :param model_name_or_path: name or path of the model to load
        :param cutoff_len: max length of sentences
        :param template: template file to use to create prompts
    """
    def __init__(self, model_name_or_path, cutoff_len = 512, template = "mentalbot"):
        self.model_name_or_path = model_name_or_path

        self.model = BloomForCausalLM.from_pretrained(model_name_or_path)
        self.tokenizer = BloomTokenizerFast.from_pretrained(model_name_or_path)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.tokenizer.padding_side = "left" # Allow batched inference

        self.cutoff_len = cutoff_len

        self.prompter = Prompter(template)

    def split_train_val_sets(self, df, val_set_size=200):
        """ Generate train, test and validation sets from dataframe 
        
        :param df: dataframe
        :return: train, val sets
        """
        # split dataset into separate training and validation sets
        train_val = df["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )

        # create prompts from the loaded dataset and tokenize them
        train_dataset = (
            train_val["train"].map(self.generate_and_tokenize_prompt)
        )
        val_dataset = (
            train_val["test"].map(self.generate_and_tokenize_prompt)
        )
        return train_dataset, val_dataset
    
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
        # Set prompt
        full_prompt = self.prompter.generate_prompt(data_point["prompt"],data_point["completion"])
        tokenized_full_prompt = self.tokenize(full_prompt)
        return tokenized_full_prompt
    
    def train(self, dataset, output_dir, epochs=1, batch_size=1, lr=5e-5):
        train_dataset, val_dataset = self.split_train_val_sets(dataset)

        warmup_steps = int(len(train_dataset)*epochs/batch_size*0.1) #10% of train data
        
        training_args = TrainingArguments(
            output_dir=output_dir,          # output directory
            num_train_epochs=epochs,         # total number of training epochs
            per_device_train_batch_size=batch_size,  # batch size per device during training
            per_device_eval_batch_size=batch_size,   # batch size for evaluation
            learning_rate=lr,               # learning rate
            warmup_steps=warmup_steps,       # number of warmup steps for learning rate scheduler
            #weight_decay=0.01,              # strength of weight decay
            logging_dir='./logs',            # directory for storing logs
            logging_steps=10,
            #save_steps=1000,                  # after # steps model is saved
            evaluation_strategy='steps',
            optim="adamw_torch",
            eval_steps=1000,                  # Number of update steps between two evaluations.
            fp16=True,                       # whether to use floating point 16 for training
            fp16_opt_level="O1",             # see apex AMP optimization level for detail
            load_best_model_at_end=True,
            #metric_for_best_model='eval_bleu',
        )

        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            #compute_metrics=compute_metrics,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator
        )

        train_result = trainer.train()
        print(trainer.evaluate())
        #trainer.save_model()  # Saves the tokenizer too for easy upload

        # Save the model and tokenizer
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

    def hyperparameter_search(self, dataset):
        train_dataset, val_dataset = self.split_train_val_sets(dataset)

        # llm hyperparams
        group_by_length = True  # faster, but produces an odd training loss curve
        
        training_args = TrainingArguments(
            output_dir='./logs',            # directory for storing output
            warmup_steps=10,       # number of warmup steps for learning rate scheduler
            #weight_decay=0.01,              # strength of weight decay
            logging_dir='./logs',            # directory for storing logs
            logging_steps=1,
            evaluation_strategy='steps',
            optim="adamw_torch",
            eval_steps=10,                  # Number of update steps between two evaluations.
            fp16=True,                       # whether to use floating point 16 for training
            fp16_opt_level="O1",             # see apex AMP optimization level for detail
            group_by_length=group_by_length,
        )

        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        )
        
        def model_init():
            return BloomForCausalLM.from_pretrained(self.model_name_or_path)
        
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
        trainer.hyperparameter_search(
            direction="minimize", 
            backend="ray", 
            n_trials=10, # number of trials
            hp_space=my_hp_space
        )

    def generate_response(self, input_text, max_length=1000, top_p=0.9):
        self.model = self.model.eval()

        # Set prompt
        prompt = self.prompter.generate_prompt(input_text)

        input_encodings = self.tokenizer(prompt, return_tensors='pt')
        input_ids = input_encodings['input_ids'].to(self.model.device)
        attention_mask = input_encodings['attention_mask'].to(self.model.device)

        # Use model.generate() to generate the response
        response = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            top_p=top_p,
            do_sample=True,
        )

        # Decode the response from the model back into text
        decoded_output = self.tokenizer.decode(response[0][ : -1])
        response = self.prompter.get_response(decoded_output)

        return response

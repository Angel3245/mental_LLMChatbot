import json
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

import logging
logging.basicConfig(
        format="%(asctime)s — %(levelname)s — %(name)s — %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

class GPT2FineTuner:
    def __init__(self, model_name_or_path, cache_dir='./cache'):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir)
        self.model = GPT2LMHeadModel.from_pretrained(model_name_or_path, cache_dir=cache_dir)
        self.cache_dir = cache_dir
        
    def load_dataset(self, dataset_path):
        self.dataset = TextDataset(
            tokenizer=self.tokenizer,
            file_path=dataset_path,
            block_size=512,
        )
        
    def fine_tune(self, output_dir='./output', num_train_epochs=3, per_device_train_batch_size=16, learning_rate=1e-4):
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            save_total_limit=2,
            save_steps=1000,
            prediction_loss_only=True,
            logging_steps=5000,
            logging_first_step=True,
            learning_rate=learning_rate,
            overwrite_output_dir=True
        )
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)
        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=self.dataset
        )
        trainer.train()
        
    def generate_text(self, prompt="", max_length=100):
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.model.device)
        output = self.model.generate(input_ids=input_ids, max_length=max_length)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)
    
    def evaluate(self, eval_dataset_path, per_device_eval_batch_size=16):
        self.model.eval()
        eval_dataset = TextDataset(eval_dataset_path, self.tokenizer)
        eval_dataloader = torch.utils.data.DataLoader(
            eval_dataset, 
            batch_size=per_device_eval_batch_size, 
            num_workers=4
        )
        eval_loss = 0.0
        eval_steps = 0
        for inputs, labels in eval_dataloader:
            inputs = inputs.to(self.model.device)
            labels = labels.to(self.model.device)
            with torch.no_grad():
                outputs = self.model(inputs, labels=labels)
                loss = outputs[0]
                eval_loss += loss.item()
                eval_steps += 1
        return eval_loss / eval_steps

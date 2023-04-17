import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from dataset import ChatbotDataset

import logging
logging.basicConfig(
        format="%(asctime)s — %(levelname)s — %(name)s — %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

class ChatbotTrainer:
    def __init__(self, model_name):
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.config.pad_token_id = self.model.config.eos_token_id

        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def split_train_val_sets(self, df):
        """ Generate train, test and validation sets from dataframe 
        
        :param df: dataframe
        :return: train, test, val sets
        """
        train, val = train_test_split(df, test_size=0.1)
        return train, val
    
    def train(self, dataset, output_dir, epochs=3, batch_size=4, lr=5e-5):
        train_dataframe, val_dataframe = self.split_train_val_sets(dataset)

        train_dataset = ChatbotDataset(train_dataframe, self.tokenizer)
        val_dataset = ChatbotDataset(val_dataframe, self.tokenizer)

        training_args = TrainingArguments(
            output_dir=output_dir,          # output directory
            num_train_epochs=epochs,         # total number of training epochs
            per_device_train_batch_size=batch_size,  # batch size per device during training
            per_device_eval_batch_size=batch_size,   # batch size for evaluation
            learning_rate=lr,               # learning rate
            warmup_steps=0,                 # number of warmup steps for learning rate scheduler
            weight_decay=0.01,              # strength of weight decay
            logging_dir='./logs',            # directory for storing logs
            logging_steps=10,
            save_steps=5000,
            evaluation_strategy='steps',
            eval_steps=5000,
            load_best_model_at_end=True,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=lambda data: {'input_ids': torch.stack([item['input_ids'] for item in data]),
                                        'attention_mask': torch.stack([item['attention_mask'] for item in data]),
                                        'labels': torch.stack([item['labels'] for item in data])},
        )

        trainer.train()

        # Save the trained model
        self.model.save_pretrained(output_dir)

    def generate_response(self, input_text, max_length=30, top_p=0.9):
        input_encodings = self.tokenizer(input_text, truncation=True, padding='max_length', max_length=512, return_tensors='pt')
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
        response_text = self.tokenizer.decode(response[0], skip_special_tokens=True)

        return response_text

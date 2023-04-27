import torch
import numpy as np
import evaluate
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from rouge_score import rouge_scorer
from train_dataset import ChatbotDataset
from shared.model_classes import MODEL_CLASSES

import logging
logging.basicConfig(
        format="%(asctime)s — %(levelname)s — %(name)s — %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

class ChatbotTrainer:
    def __init__(self, model_name):
        model_class, tokenizer_class, model_name_or_path = MODEL_CLASSES[model_name]

        self.model = model_class.from_pretrained(model_name_or_path)

        self.tokenizer = tokenizer_class.from_pretrained(model_name_or_path)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.tokenizer.padding_side = "left" # Allow batched inference
        self.tokenizer.sep_token = "<sep>"

    def split_train_val_sets(self, df):
        """ Generate train, test and validation sets from dataframe 
        
        :param df: dataframe
        :return: train, test, val sets
        """
        train, val = train_test_split(df, test_size=0.1)
        return train, val
    
    def train(self, dataset, output_dir, epochs=1, batch_size=1, lr=5e-5):
        train_dataframe, val_dataframe = self.split_train_val_sets(dataset)

        train_dataset = ChatbotDataset(train_dataframe, self.tokenizer)
        val_dataset = ChatbotDataset(val_dataframe, self.tokenizer)

        training_args = TrainingArguments(
            output_dir=output_dir,          # output directory
            num_train_epochs=epochs,         # total number of training epochs
            per_device_train_batch_size=batch_size,  # batch size per device during training
            per_device_eval_batch_size=batch_size,   # batch size for evaluation
            learning_rate=lr,               # learning rate
            #warmup_steps=500,                 # number of warmup steps for learning rate scheduler
            #weight_decay=0.01,              # strength of weight decay
            logging_dir='./logs',            # directory for storing logs
            logging_steps=10,
            save_steps=500,                  # after # steps model is saved
            evaluation_strategy='steps',
            eval_steps=100,                  # Number of update steps between two evaluations.
            fp16=True,                       # whether to use floating point 16 for training
            fp16_opt_level="O1",             # see apex AMP optimization level for detail
            load_best_model_at_end=True,
            metric_for_best_model='eval_bleu',
        )

        metric_bleu = evaluate.load("bleu")

        def compute_metrics(eval_preds):
            logits, labels = eval_preds

            predictions = logits.argmax(axis=-1)
            
            preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
            labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

            bleu = metric_bleu.compute(predictions=preds, references=labels)

            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

            # Compute rouge score between prediction and target
            rouge_scores = []
            for i in range(len(preds)):
                prediction = preds[i]
                target = labels[i]
                scores = scorer.score(target, prediction)
                rouge_scores.append(scores)

            # Compute average rouge scores
            rouge1 = np.mean([score['rouge1'].fmeasure for score in rouge_scores])
            rouge2 = np.mean([score['rouge2'].fmeasure for score in rouge_scores])
            rougeL = np.mean([score['rougeL'].fmeasure for score in rouge_scores])

            return {"eval_bleu": bleu, 'rouge1': rouge1, 'rouge2': rouge2, 'rougeL': rougeL}
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            #compute_metrics=compute_metrics,
            train_dataset=train_dataset,
            eval_dataset=val_dataset
        )

        train_result = trainer.train()
        #print(trainer.evaluate())
        #trainer.save_model()  # Saves the tokenizer too for easy upload

        # Save the model and tokenizer
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)


    def generate_response(self, input_text, max_length=1000, top_p=0.9):
        # Set prompt
        input_text = self.tokenizer.bos_token + input_text + self.tokenizer.sep_token

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
        response_text = self.tokenizer.decode(response[0][ : -1], skip_special_tokens=False).split(self.tokenizer.sep_token)[1]

        return response_text

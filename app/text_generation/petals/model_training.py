import torch
import numpy as np
import evaluate
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from rouge_score import rouge_scorer
from train_dataset import PetalsDataset

from transformers import BloomTokenizerFast
from petals import DistributedBloomForCausalLM

import logging
logging.basicConfig(
        format="%(asctime)s — %(levelname)s — %(name)s — %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

class PetalsTrainer:
    def __init__(self, model_name="bigscience/bloom-petals"):
        self.model = DistributedBloomForCausalLM.from_pretrained(model_name, tuning_mode="ptune", pre_seq_len=16)

        self.tokenizer = BloomTokenizerFast.from_pretrained(model_name)
        self.tokenizer.model_max_length = 256

        self.tokenizer.padding_side = "left" # Allow batched inference

    def split_train_val_sets(self, df):
        """ Generate train, test and validation sets from dataframe 
        
        :param df: dataframe
        :return: train, test, val sets
        """
        train, val = train_test_split(df, test_size=0.1)
        return train, val
    
    def train(self, dataset, output_dir, epochs=1, batch_size=8, lr=1e-2):
        train_dataframe, val_dataframe = self.split_train_val_sets(dataset)

        train_dataset = PetalsDataset(train_dataframe, self.tokenizer)
        val_dataset = PetalsDataset(val_dataframe, self.tokenizer)

        #warmup_steps = int(len(train_dataset)*epochs/batch_size*0.1) #10% of train data
        
        training_args = TrainingArguments(
            output_dir=output_dir,          # output directory
            num_train_epochs=epochs,         # total number of training epochs
            per_device_train_batch_size=batch_size,  # batch size per device during training
            per_device_eval_batch_size=batch_size,   # batch size for evaluation
            learning_rate=lr,               # learning rate
            warmup_steps=0,       # number of warmup steps for learning rate scheduler
            weight_decay=0.0,              # strength of weight decay
            logging_dir='./logs',            # directory for storing logs
            logging_steps=10,
            save_steps=1000,                  # after # steps model is saved
            evaluation_strategy='steps',
            optim="adamw_torch",
            eval_steps=1000,                  # Number of update steps between two evaluations.
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

            return {"eval_bleu": bleu["bleu"], 'rouge1': rouge1, 'rouge2': rouge2, 'rougeL': rougeL}
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            compute_metrics=compute_metrics,
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
    
    def generate_response(self, input_text, max_length=1000, temperature=0.6, top_k=100):
        # Set prompt: <bos> input <sep>
        input_ids = self.tokenizer(f"User:{input_text}\nBot:", return_tensors='pt')['input_ids'].to(self.model.device)

        response_text = ""

        # Use model.generate() to generate the response
        with self.model.inference_session(max_length=512) as sess:
            while True:
                outputs = self.model.generate(
                    input_ids,
                    temperature=temperature,
                    top_k=top_k,
                    max_new_tokens=1,
                    do_sample=True,
                    session=sess,
                )

                bloom_answer_token = self.tokenizer.decode(outputs[0, -1:])
                response_text += bloom_answer_token
                if bloom_answer_token == "\n":
                    break

                input_ids = None

        return response_text

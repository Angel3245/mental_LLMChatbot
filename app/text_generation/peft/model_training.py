import torch
import os, sys
import numpy as np
import evaluate
from transformers import Trainer, TrainingArguments, GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from sklearn.model_selection import train_test_split
from rouge_score import rouge_scorer
from train_dataset import LlamaDataset

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

class PeftTrainer:
    def __init__(self, model_name, cutoff_len = 512):
        device_map = "auto"

        self.model = LlamaForCausalLM.from_pretrained(
            model_name,
            load_in_8bit=True,
            torch_dtype=torch.float16,
            device_map=device_map,
        )

        self.tokenizer = LlamaTokenizer.from_pretrained(model_name)

        self.tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
        self.tokenizer.padding_side = "left" # Allow batched inference

        self.model = prepare_model_for_int8_training(self.model)

        # training hyperparams
        self.cutoff_len = cutoff_len

        # lora hyperparams
        lora_r = 8
        lora_alpha = 16
        lora_dropout = 0.05
        lora_target_modules = [
            "q_proj",
            "v_proj",
        ]

        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, config)

    def train(self, dataset, output_dir, epochs=1, batch_size=4, lr=2e-5, val_set_size=0.1):
        train_dataframe, val_dataframe = train_test_split(dataset,test_size=val_set_size)

        train_dataset = LlamaDataset(train_dataframe, self.tokenizer)
        val_dataset = LlamaDataset(val_dataframe, self.tokenizer)

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
            logging_steps=10,
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

        #metrics = train_result.metrics
        #trainer.log_metrics("train", metrics)
        #trainer.save_metrics("train", metrics)

    def generate_response(self, input_text, max_new_tokens=256, temperature=0.1, top_p=0.9, top_k=40, num_beams=1, repetition_penalty=1.1):

        self.model.eval()

        # Set prompt
        prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n" + input_text + "\n\n### Response:\n"

        input_encodings = self.tokenizer(prompt, return_tensors='pt')
        input_ids = input_encodings['input_ids'].to(self.model.device)

        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty
        )
        
        with torch.inference_mode():
            # Use model.generate() to generate the response
            response = self.model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                #do_sample=True,
                max_new_tokens=max_new_tokens,
            )

        # Decode the response from the model back into text
        decoded_output = self.tokenizer.decode(response.sequences[0])
        response = decoded_output.split("### Response:")[1].strip()

        return response

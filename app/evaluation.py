import argparse
import sys
from pathlib import Path
from shared import *
from text_generation.gpt2 import GPT2Trainer
from text_generation.bloom import BloomPeftTrainer
from text_generation.peft import PeftTrainer
if sys.platform != "win32":
    from text_generation.petals import PetalsTrainer
from transformation import *
import pandas as pd
from transformers import GPT2Tokenizer
import random
from datasets import load_dataset

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--option", type=str, help="select an option", required=True)
    parser.add_argument("-m", "--model", type=str, help="select a model pretrained: gpt2, bloom, petals, peft. Default: gpt2", default='gpt2')
    parser.add_argument("-d", "--dataset", type=str, help="select a dataset. Default: MentalKnowledge", default="MentalKnowledge")
    parser.add_argument("-t", "--template", type=str, help="select a template to create prompts. See /file/templates")
    parser.add_argument("-b", "--base_model", type=str, help="select a model to load from huggingface")
    args = parser.parse_args()

    path = Path.cwd()

    if args.option == "evaluate":
        # python app\evaluation.py -o evaluate -m gpt2 -b gpt2
        model_name = args.model

        test_filepath = F"{str(path)}/file/test/test_inputs.json"
        output_path = F"{str(path)}/file/evaluation/MentalKnowledge/"+model_name

        dataset = load_dataset("json", data_files=test_filepath)
        print("Test dataset:",dataset["train"])

        #dataset = random.sample(list(dataset), 30)

        # Load model from disk
        model_path = F"{str(path)}/output/MentalKnowledge/"+model_name+"/"+args.base_model
        print("Loading model from",model_path)

        if(model_name == "gpt2"):
            model = GPT2Trainer(model_path)
        elif(model_name == "bloom"):
            #model = BloomTrainer(model_path)
            model = BloomPeftTrainer(model_path=model_path)
        elif(model_name == "petals"):
            model = PetalsTrainer(model_path)
        elif(model_name == "peft"):
            model = PeftTrainer(model_path, args.base_model, is_peft=True)
        else:
            raise ValueError('model ' + model_name + ' not exist')

        model.evaluation(dataset, output_path)

        print("Evaluation results dumped to",output_path)


    ###

    print("PROGRAM FINISHED")
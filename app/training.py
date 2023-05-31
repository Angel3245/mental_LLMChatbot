import argparse
import sys
from pathlib import Path
from shared import *
from text_generation.gpt2 import GPT2Trainer
from text_generation.bloom import BloomTrainer, BloomPeftTrainer
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
    parser.add_argument("-m", "--model", type=str, help="select a model", default='GPT2')
    parser.add_argument("-b", "--base_model", type=str, help="select a model to load from huggingface")
    parser.add_argument("-d", "--dataset", type=str, help="select a dataset", default="MentalKnowledge")
    parser.add_argument("-e", "--epochs", type=int, help="select a number of epochs for training")
    parser.add_argument("-batch", "--batch_size", type=int, help="select a batch size value")
    parser.add_argument("-lr", "--learning_rate", type=float, help="select a learning rate value")
    parser.add_argument("-neg", "--neg_type", type=str, help="select a negative type")
    parser.add_argument("-loss", "--loss_type", type=str, help="select a loss_type")
    parser.add_argument("-db", "--database", type=str, help="select a database")
    parser.add_argument("-f", "--file", type=str, help="select a file")
    args = parser.parse_args()

    path = Path.cwd()

    if args.option == "finetune_model":
        # python app\training.py -o finetune_model -m gpt2 -b gpt2
        dataset_filepath = F"{str(path)}/file/data/MentalKnowledge/input_label_pairs.json"
        dataset = load_dataset("json", data_files=dataset_filepath)
        print("Train dataset:",dataset["train"])

        #dataset = random.sample(list(dataset), 30)

        model_name = args.model
        output_path = F"{str(path)}/output/MentalKnowledge/"+model_name+"/"+args.base_model

        if(model_name == "gpt2"):
            model = GPT2Trainer(args.base_model) #gpt2
        elif(model_name == "bloom"):
            model = BloomPeftTrainer(None,args.base_model) #bigscience/bloom-1b7
        elif(model_name == "petals"):
            model = PetalsTrainer(args.base_model) #bigscience/bloom-7b1-petals
        elif(model_name == "peft"):
            model = PeftTrainer(None,args.base_model) #decapoda-research/llama-7b-hf, chavinlo/alpaca-native
        else:
            raise ValueError('model ' + model_name + ' not exist')

        model.train(dataset, output_path)
        print(model.generate_response("Where can I find self help materials for anxiety?"))

    if args.option == "hyperparameter_search":
        # python app\training.py -o hyperparameter_search -m gpt2
        dataset_filepath = F"{str(path)}/file/data/MentalKnowledge/input_label_pairs.json"
        dataset = load_dataset("json", data_files=dataset_filepath)
        print("Train dataset:",dataset["train"])

        #dataset = random.sample(list(dataset), 30)

        model_name = args.model

        if(model_name == "gpt2"):
            model = GPT2Trainer(model_name)
        elif(model_name == "bloom"):
            model = BloomTrainer("bigscience/bloom-560m")
        elif(model_name == "petals"):
            model = PetalsTrainer("bigscience/bloom-7b1-petals")
        elif(model_name == "peft"):
            model = PeftTrainer("decapoda-research/llama-7b-hf")
        else:
            raise ValueError('model ' + model_name + ' not exist')

        model.hyperparameter_search(dataset)

    # Create CSV with results from several hyperparameter configurations
    '''
    if args.option == "hyperparameter_tuning":
        # python app\training.py -o hyperparameter_tuning -d MentalKnowledge -m gpt2
        batch_sizes = [16, 32]
        learning_rate = [5e-5, 3e-5, 2e-5]
        adam_epsilons = [1e-8]
        epochs = [2, 3, 4]
        model_list = [str(item) for item in args.model.split(',')]

        # Creating Empty DataFrame
        df_triplet = pd.DataFrame()
        df_softmax = pd.DataFrame()
    
        if(args.dataset == "MentalKnowledge"):
            data_path = F"{str(path)}/file/data/MentalKnowledge"
            output_path = F"{str(path)}/output/MentalKnowledge"

        for model_name in model_list:
            for epoch_value in epochs:
                for batch_value in batch_sizes:
                    for lr_value in learning_rate:
                        model_output_path = output_path

                        result = get_finetune_results(data_path,model_output_path,model_name,epoch_value,batch_value,lr_value)

                        df_triplet = pd.concat([df_triplet, result["triplet"]], axis=0)
                        df_softmax = pd.concat([df_softmax, result["softmax"]], axis=0)

        # Dump data to csv file
        make_dirs(output_path+"/evaluation_results")
        df_triplet.to_csv(output_path+"/evaluation_results/triplet_results.csv")
        df_softmax.to_csv(output_path+"/evaluation_results/softmax_results.csv")
    '''


    ###

    print("PROGRAM FINISHED")
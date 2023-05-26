import argparse
from pathlib import Path
from shared import *
from text_generation.gpt2 import *
from text_generation.peft import *
#from text_generation.petals import *
from transformation import *
import pandas as pd
from transformers import GPT2Tokenizer
import random

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--option", type=str, help="select an option", required=True)
    parser.add_argument("-m", "--model", type=str, help="select a model", default='GPT2')
    parser.add_argument("-d", "--dataset", type=str, help="select a dataset", default="MentalKnowledge")
    parser.add_argument("-e", "--epochs", type=int, help="select a number of epochs for training")
    parser.add_argument("-b", "--batch_size", type=int, help="select a batch size value")
    parser.add_argument("-lr", "--learning_rate", type=float, help="select a learning rate value")
    parser.add_argument("-neg", "--neg_type", type=str, help="select a negative type")
    parser.add_argument("-loss", "--loss_type", type=str, help="select a loss_type")
    parser.add_argument("-db", "--database", type=str, help="select a database")
    parser.add_argument("-f", "--file", type=str, help="select a file")
    args = parser.parse_args()

    path = Path.cwd()

    if args.option == "evaluate":
        # python app\evaluate.py -o evaluate -m gpt2
        test_filepath = F"{str(path)}/file/test/test_inputs.txt"

        with open(test_filepath, "r") as test_file:
            test_inputs = test_file.readlines()

        #dataset = random.sample(list(dataset), 30)

        model_name = args.model
        output_path = F"{str(path)}/file/evaluation/MentalKnowledge/"+model_name

        if(model_name == "gpt2"):
            model = GPT2Trainer(model_name)
        elif(model_name == "petals"):
            model = PetalsTrainer("bigscience/bloom-7b1-petals")
        elif(model_name == "peft"):
            model = PeftTrainer("decapoda-research/llama-7b-hf")
        else:
            raise ValueError('model ' + model_name + ' not exist')

        model.evaluation(test_inputs, output_path)

        print("Evaluation results dumped to",output_path)

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
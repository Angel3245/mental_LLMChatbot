import argparse
from pathlib import Path
from DB.connect import database_connect
from shared import *
from model import *
from text_generation import *
from transformation import *
import pandas as pd
from transformers import GPT2Tokenizer

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

    if args.option == "finetune_model":
        # python app\training.py -o finetune_model -m GPT2
        dataset_filepath = F"{str(path)}/file/data/MentalKnowledge/input_label_pairs.json"
        output_path = F"{str(path)}/output/MentalKnowledge"
        model_name = args.model

        if(model_name == "GPT2"):
            model = GPT2FineTuner("gpt2")
        else:
            raise ValueError('model' + model_name + 'not exist')

        model.load_dataset(dataset_filepath)
        model.fine_tune(output_path)

    # Finetune model
    if args.option == "model_training":
        # python app\training.py -o model_training -d MentalKnowledge -m GPT-2 -e 2 -b 32 -lr 3e-05 -neg simple -loss softmax
        model_name = args.model
        epoch_value = args.epochs
        batch_value = args.batch_size
        lr_value = args.learning_rate
        neg_type = args.neg_type
        loss_type = args.loss_type
    
        if(args.dataset == "MentalKnowledge"):
            data_path = F"{str(path)}/file/data/MentalKnowledge"
            output_path = F"{str(path)}/output/MentalKnowledge"

        else:
            raise ValueError("Dataset not selected")

        model_output_path = output_path+"_"+model_name

        model_training(data_path,model_output_path,model_name,neg_type,loss_type, epochs=epoch_value,batch_size=batch_value,learning_rate=lr_value)

    # Create CSV with results
    if args.option == "get_finetune_results":
        # python app\training.py -o get_finetune_results -d MentalKnowledge -m GPT-2
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



    ###

    print("PROGRAM FINISHED")
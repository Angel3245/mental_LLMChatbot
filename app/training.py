import argparse
import sys
from pathlib import Path
from text_generation.model_classes import ModelDispatcher
from shared import *
from text_generation.gpt2 import GPT2Trainer
from text_generation.bloom import BloomTrainer, BloomPeftTrainer
from text_generation.llama import LlamaPeftTrainer
if sys.platform != "win32":
    from text_generation.petals import PetalsTrainer
from datasets import load_dataset

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--option", type=str, help="select an option", required=True)
    parser.add_argument("-m", "--model", type=str, help="select a model name to load", default='gpt2', required=True)
    parser.add_argument("-d", "--dataset", type=str, help="select a dataset", default="MentalKnowledge")

    args = parser.parse_args()

    path = Path.cwd()

    # Trainer classes
    trainers = {
        "gpt2": GPT2Trainer,
        "bloom": BloomPeftTrainer,
        #"petals": PetalsTrainer,
        "llama": LlamaPeftTrainer
    }
    
    # Train a language model using a Trainer 
    if args.option == "finetune_model":
        # python app\training.py -o finetune_model -m gpt2

        # Load dataset
        dataset_filepath = F"{str(path)}/file/data/"+args.dataset+"/input_label_pairs.json"
        dataset = load_dataset("json", data_files=dataset_filepath)
        print("Train dataset:",dataset["train"])

        # Get model_type from dispatcher
        model_name = args.model
        model_type = ModelDispatcher.get_model_type(model_name)
        
        output_path = F"{str(path)}/output/MentalKnowledge/"+model_type+"/"+model_name

        # Get model trainer class from type
        model = trainers[model_type](model_name)

        # Train model with input dataset
        model.train(dataset, output_path)

        # Test inputs
        print("Where can I find self help materials for anxiety?")
        print(model.generate_response("Where can I find self help materials for anxiety?"))
        print("What is depression?")
        print(model.generate_response("What is depression?"))

    # Search hyperparameters using Ray Tune
    if args.option == "hyperparameter_search":
        # python app\training.py -o hyperparameter_search -m gpt2
        dataset_filepath = F"{str(path)}/file/data/"+args.dataset+"/input_label_pairs.json"
        dataset = load_dataset("json", data_files=dataset_filepath)
        print("Train dataset:",dataset["train"])

        # Get model_type from dispatcher
        model_name = args.model
        model_type = ModelDispatcher.get_model_type(model_name)
        
        output_path = F"{str(path)}/output/"+args.dataset+"/"+model_type+"/"+model_name

        # Get model trainer class from type
        model = trainers[model_type](model_name)

        # Look for the best hyperparameters
        model.hyperparameter_search(dataset)

    print("PROGRAM FINISHED")
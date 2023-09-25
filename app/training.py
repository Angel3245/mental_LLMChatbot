# Copyright (C) 2023  Jose Ángel Pérez Garrido
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

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

    parser = argparse.ArgumentParser(prog='Training',
                    description='Train a pretrained model',
                    epilog='Jose Angel Perez Garrido - 2023')
    parser.add_argument("-o", "--option", type=str, help="select an option: finetune_model -> Train a language model using a Trainer; hyperparameter_search -> Search best hyperparameters using Ray Tune", required=True)
    parser.add_argument("-m", "--model", type=str, help="select a pretrained model to load. Supported models: "+str(ModelDispatcher.get_supported_types()), required=True)
    parser.add_argument("-d", "--dataset", type=str, help="select a dataset. (default: MentalKnowledge)", default="MentalKnowledge")

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

        print("Training a",model_type,"model")
        
        output_path = F"{str(path)}/output/"+args.dataset+"/"+model_type+"/"+model_name

        # Get model trainer class from type
        model = trainers[model_type](model_name)

        # Train model with input dataset
        model.train(dataset, output_path)

        print("Model trained saved to",output_path)

        # Test inputs
        print("\nTest inputs:\n")
        print("Where can I find self help materials for anxiety?")
        print(model.generate_response("Where can I find self help materials for anxiety?"))
        print("What is depression?")
        print(model.generate_response("What is depression?"))

    # Search best hyperparameters using Ray Tune
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
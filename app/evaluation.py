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
import sys, csv, evaluate
from pathlib import Path
from shared import *
from text_generation.model_classes import ModelDispatcher
from text_generation.gpt3 import GPT3TextGenerator
from text_generation.gpt2 import GPT2TextGenerator
from text_generation.bloom import BloomTextGenerator, BloomPeftTextGenerator
from text_generation.llama import LlamaPeftTextGenerator
if sys.platform != "win32":
    from text_generation.petals import PetalsTextGenerator

from datasets import load_dataset

if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog='Evaluation',
                    description='Evaluate the results of a LLM',
                    epilog='Jose Angel Perez Garrido - 2023')
    parser.add_argument("-o", "--option", type=str, help="select an option: evaluate -> test LLM using a test dataset; ask -> test LLM asking questions", required=True)
    parser.add_argument("-m", "--model", type=str, help="select a pretrained model to load. Supported models: "+str(ModelDispatcher.get_supported_types())+" (default: gpt2)", default='gpt2')
    parser.add_argument("-d", "--dataset", type=str, help="select a dataset. (default: MentalKnowledge)", default="MentalKnowledge")
    #parser.add_argument("-t", "--template", type=str, help="select a template to create prompts. See /file/templates")
    args = parser.parse_args()

    path = Path.cwd()

    # Text generator classes
    text_generators = {
        "gpt2": GPT2TextGenerator,
        "bloom": BloomPeftTextGenerator,
        #"petals": PetalsTextGenerator,
        "llama": LlamaPeftTextGenerator
    }

    # Create CSV with results from test cases
    if args.option == "evaluate":
        # python app\evaluation.py -o evaluate -m gpt2

        # Get model_type from dispatcher
        model_name = args.model
        model_type = ModelDispatcher.get_model_type(model_name)

        test_filepath = F"{str(path)}/file/test/test_inputs.json"
        output_path = F"{str(path)}/file/evaluation/"+args.dataset+"/"+model_type+"_"+model_name

        # Load test dataset
        dataset = load_dataset("json", data_files=test_filepath)
        test_inputs = dataset["train"]
        print("Test dataset:",test_inputs)

        #dataset = random.sample(list(dataset), 30)

        # Get model text_generator class from type
        text_generator = text_generators[model_type](model_name)

        # Load automatic metrics
        bleu_metric = evaluate.load("bleu")
        rouge_metric = evaluate.load("rouge")

        # Create CSV with evaluation results
        make_dirs(output_path)
        with open(output_path+"/evaluation.csv", 'w', encoding="UTF8") as csv_file:
            writer = csv.writer(csv_file, delimiter=",")
            writer.writerow(["Input","Response","Bleu-1","Rouge-1"])

        for input_text in test_inputs:
            response = text_generator.generate_response(input_text["input"])

            with open(output_path+"/evaluation.csv", 'a', encoding="UTF8") as csv_file:
                writer = csv.writer(csv_file, delimiter=",")
                writer.writerow([input_text["input"],response, round(bleu_metric.compute(predictions=[response],references=[input_text["output_expected"]])['precisions'][0] ,2), round(rouge_metric.compute(predictions=[response],references=[input_text["output_expected"]])['rouge1'] ,2)])

        print("Evaluation results dumped to",output_path)

    # Test text_generation models
    if args.option == "ask":
        # python app\evaluation.py -o ask -m gpt2

        model_name = args.model

        # Get model_type from dispatcher
        model_type = ModelDispatcher.get_model_type(model_name)

        # Load model from disk
        model_path = F"{str(path)}/output/"+args.dataset+"/"+model_type+"/"+model_name
        print("Loading model from",model_path)

        # Text generator classes
        text_generators = {
            "gpt2": GPT2TextGenerator,
            "gpt3": GPT3TextGenerator,
            "bloom": BloomPeftTextGenerator,
            #"petals": PetalsTextGenerator,
            "llama": LlamaPeftTextGenerator
        }

        # Get model text_generator class from type
        text_generator = text_generators[model_type](model_path)

        # Ask questions to chatbot and create responses
        while True:
            user_input = input("User: ")
            if user_input.lower() == 'exit':
                break
            response = text_generator.generate_response(user_input)
            print(f"Chatbot: {response}")

    print("PROGRAM FINISHED")
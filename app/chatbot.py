import argparse
import sys
from pathlib import Path
from text_generation.model_classes import ModelDispatcher
from shared import *
from text_generation.gpt2 import GPT2TextGenerator
from text_generation.gpt3 import GPT3TextGenerator
from text_generation.bloom import BloomTextGenerator, BloomPeftTextGenerator
from text_generation.llama import LlamaPeftTextGenerator
if sys.platform != "win32":
    from text_generation.petals import PetalsTextGenerator
from clean_data import clean_sentence

if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog='Chatbot',
                    description='Run a chatbot using a LLM',
                    epilog='Jose Angel Perez Garrido - 2023')
    parser.add_argument("-o", "--option", type=str, help="select an option: cli -> run a chatbot in command line view. (default: cli)", default="cli")
    parser.add_argument("-m", "--model", type=str, help="select a pretrained model to load. Supported models: "+str(ModelDispatcher.get_supported_types())+" (default: gpt2)", default='gpt2')
    parser.add_argument("-d", "--dataset", type=str, help="select a dataset. (default: MentalKnowledge)", default="MentalKnowledge")
    #parser.add_argument("-t", "--template", type=str, help="select a template file to create prompts. See /file/templates")
    args = parser.parse_args()

    path = Path.cwd()

    if args.option == "cli":
        # python app\chatbot.py -o ask -m gpt2

        model_name = args.model

        # Get model_type from dispatcher
        model_type = ModelDispatcher.get_model_type(model_name)

        # Load model from disk
        model_path = F"{str(path)}/output/MentalKnowledge/"+model_type+"/"+model_name
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
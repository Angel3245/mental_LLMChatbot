import argparse
from pathlib import Path
from shared import *
from text_generation.bloom import BloomPeftTextGenerator

from clean_data import clean

if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog='Chatbot',
                    description='Run a chatbot using a LLM',
                    epilog='Jose Angel Perez Garrido - 2023')
    parser.add_argument("-o", "--option", type=str, help="select an option: cli -> run a chatbot in command line view. (default: cli)", default="cli")
    #parser.add_argument("-m", "--model", type=str, help="select a pretrained model to load. Supported models: "+str(ModelDispatcher.get_supported_types())+" (default: gpt2)", default='gpt2')
    #parser.add_argument("-d", "--dataset", type=str, help="select a dataset. (default: MentalKnowledge)", default="MentalKnowledge")
    #parser.add_argument("-t", "--template", type=str, help="select a template file to create prompts. See /file/templates")
    args = parser.parse_args()

    path = Path.cwd()

    if args.option == "cli":
        # python app\chatbot.py -o cli

        # Load model from disk
        model_path = F"{str(path)}/file/chatbot_model"
        print("Loading chatbot model from",model_path)

        # Get BLOOM text_generation model
        text_generator = BloomPeftTextGenerator(model_path)

        # Ask questions to chatbot and create responses
        while True:
            user_input = input("User: ")
            if user_input.lower() == 'exit':
                break

            # Clean input sentence
            user_input = clean(user_input)

            # Analyze input and create predefined responses if needed
            # Check if user_input words are higher than 3
            if(len(user_input.split()) < 3):
                response = "Sorry, I did not understand you. Could you explain it better?"
            else:
                response = text_generator.generate_response(user_input)
            
            print(f"Chatbot: {response}")

    print("PROGRAM FINISHED")
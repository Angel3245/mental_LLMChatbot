import argparse
import sys
from pathlib import Path
from text_generation.gpt2 import GPT2Chatbot
from text_generation.bloom import BloomChatbot, BloomPeftChatbot
from text_generation.peft import PeftChatbot
if sys.platform != "win32":
    from text_generation.petals import PetalsChatbot
from preprocessing import preprocess

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--option", type=str, help="select an option", required=True)
    parser.add_argument("-m", "--model", type=str, help="select a model pretrained: gpt2, bloom, petals, peft. Default: gpt2", default='gpt2')
    parser.add_argument("-d", "--dataset", type=str, help="select a dataset. Default: MentalKnowledge", default="MentalKnowledge")
    parser.add_argument("-t", "--template", type=str, help="select a template to create prompts. See /file/templates")
    parser.add_argument("-b", "--base_model", type=str, help="select a model to load from huggingface")
    args = parser.parse_args()

    path = Path.cwd()

    if args.option == "ask":
        # python app\chatbot.py -o ask -m gpt2 -t chatbot_simple

        model_name = args.model

        # Load model from disk
        model_path = F"{str(path)}/output/MentalKnowledge/"+model_name+"/"+args.base_model
        print("Loading model from",model_path)

        if(model_name == "gpt2"):
            chatbot = GPT2Chatbot(model_path, args.template)
        elif(model_name == "bloom"):
            #chatbot = BloomChatbot(model_path, args.template)
            chatbot = BloomPeftChatbot(model_path, args.template)
        elif(model_name == "petals"):
            chatbot = PetalsChatbot(model_path, args.template)
        elif(model_name == "peft"):
            chatbot = PeftChatbot(model_path,args.base_model, args.template)
        else:
            raise ValueError('model ' + model_name + ' not exist')
        
        #chatbot = Chatbot("gpt2")

        # Ask questions to chatbot and create responses
        while True:
            user_input = input("Usuario: ")
            if user_input.lower() == 'salir':
                break
            response = chatbot.generate_response(user_input)
            print(f"Chatbot: {response}")

    if args.option == "gui":
        # python app\chatbot.py -o gui
        dataset = F"{str(path)}/file/datasets/Reddit_posts.csv"

        #ask_sentence(dataset)

    print("PROGRAM FINISHED")
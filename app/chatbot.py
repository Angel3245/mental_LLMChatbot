import argparse
from pathlib import Path
from text_generation.gpt2 import GPT2Chatbot
from text_generation.peft import PeftChatbot
from text_generation.petals import PetalsChatbot
from preprocessing import preprocess

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--option", type=str, help="select an option", required=True)
    parser.add_argument("-m", "--model", type=str, help="select a model", default='gpt2')
    parser.add_argument("-d", "--dataset", type=str, help="select a dataset", default="MentalKnowledge")
    parser.add_argument("-f", "--file", type=str, help="select a file")
    parser.add_argument("-t", "--text", type=str, help="introduce a mental health related text")
    args = parser.parse_args()

    path = Path.cwd()

    if args.option == "ask":
        # python app\chatbot.py -o ask -m gpt2

        model_name = args.model
        # Cargar el modelo previamente entrenado desde el disco duro
        model_path = F"{str(path)}/output/MentalKnowledge/"+model_name

        if(model_name == "gpt2"):
            chatbot = GPT2Chatbot(model_name, model_path)
        elif(model_name == "petals"):
            chatbot = PetalsChatbot(model_path)
        elif(model_name == "peft"):
            chatbot = PeftChatbot(model_path,"decapoda-research/llama-7b-hf")
        else:
            raise ValueError('model ' + model_name + ' not exist')
        
        #chatbot = Chatbot("gpt2")

        # Hacer preguntas al chatbot y mostrar las respuestas
        while True:
            user_input = input("Usuario: ")
            if user_input.lower() == 'salir':
                break
            response = chatbot.generate_response(preprocess(user_input))
            print(f"Chatbot: {response}")

    if args.option == "gui":
        # python app\chatbot.py -o gui
        dataset = F"{str(path)}/file/datasets/Reddit_posts.csv"

        #ask_sentence(dataset)

    print("PROGRAM FINISHED")
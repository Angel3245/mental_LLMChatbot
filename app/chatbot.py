import argparse
from pathlib import Path
from text_generation import Chatbot
from preprocessing import preprocess

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--option", type=str, help="select an option", required=True)
    parser.add_argument("-m", "--model", type=str, help="select a model", default='distilbert-base-uncased')
    parser.add_argument("-d", "--dataset", type=str, help="select a dataset", default="MentalFAQ")
    parser.add_argument("-f", "--file", type=str, help="select a file")
    parser.add_argument("-t", "--text", type=str, help="introduce a mental health related text")
    args = parser.parse_args()

    path = Path.cwd()

    if args.option == "ask":
        # python app\chatbot.py -o ask

        # Cargar el modelo previamente entrenado desde el disco duro
        model_path = F"{str(path)}/output/MentalKnowledge"
        chatbot = Chatbot(model_path)

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
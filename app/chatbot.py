import argparse
from pathlib import Path
from faq.Mental_Health_FAQ import *
from view import *

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--option", type=str, help="select an option", required=True)
    parser.add_argument("-m", "--model", type=str, help="select a model", default='distilbert-base-uncased')
    parser.add_argument("-d", "--dataset", type=str, help="select a dataset", default="MentalFAQ")
    parser.add_argument("-f", "--file", type=str, help="select a file")
    parser.add_argument("-t", "--text", type=str, help="introduce a mental health related text")
    args = parser.parse_args()

    path = Path.cwd()

    if args.option == "retrieve_answer":
        # python app\chatbot.py -o retrieve_answer -t "What is a mental illness?"
        top_k = 3
        dataset = 'MentalKnowledge'
        model = 'publichealthsurveillance/PHS-BERT'
        fields = ['question_answer']
        model_path = dataset+"/models/"+model

        index_name = "mentalknowledge"

        # Define model parameters
        loss_type = 'softmax'; neg_type = 'simple'; query_type = 'faq'

        # Input text
        text = args.text

        print(ranker(top_k, model_path, fields, index_name, loss_type, neg_type, query_type, text))

    if args.option == "gui":
        # python app\chatbot.py -o gui
        dataset = F"{str(path)}/file/datasets/Reddit_posts.csv"

        ask_sentence(dataset)

    print("PROGRAM FINISHED")
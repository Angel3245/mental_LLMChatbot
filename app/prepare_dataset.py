import argparse
import json
from pathlib import Path
from shared import *
from transformation import create_dataset

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--option", type=str, help="select an option", required=True)
    parser.add_argument("-m", "--model", type=str, help="select a model", default='distilbert-base-uncased')
    parser.add_argument("-d", "--dataset", type=str, help="select a dataset", default="MentalFAQ")
    parser.add_argument("-e", "--epochs", type=int, help="select a number of epochs for training")
    parser.add_argument("-b", "--batch_size", type=int, help="select a batch size value")
    parser.add_argument("-lr", "--learning_rate", type=float, help="select a learning rate value")
    parser.add_argument("-neg", "--neg_type", type=str, help="select a negative type")
    parser.add_argument("-loss", "--loss_type", type=str, help="select a loss_type")
    parser.add_argument("-db", "--database", type=str, help="select a database")
    parser.add_argument("-f", "--file", type=str, help="select a file")
    args = parser.parse_args()

    path = Path.cwd()
    
    # Parse a dataset for text generation
    if args.option == "parsing_text_generation":
        # python app\prepare_dataset.py -o parsing_text_generation -d MentalKnowledge
        if(args.dataset):
            file_path = F"{str(path)}/file/datasets/Mental_Health_FAQ.csv"
            posts_path = F"{str(path)}/file/datasets/Reddit_posts.csv"
            comments_path = F"{str(path)}/file/datasets/Reddit_comments.csv"
            output_path = F"{str(path)}/file/data/"+args.dataset

            dataset_processed = create_dataset(file_path, posts_path, comments_path)

            # Dump data to json file
            make_dirs(output_path)
            dump_to_json(dataset_processed, output_path+'/input_label_pairs.json', sort_keys=False)

            print("Dataset",args.dataset, "created.")
        else:
            print("Dataset not selected")

    if args.option == "prepare_data":
        # python app\prepare_dataset.py -o prepare_data -d MentalKnowledge
        if(args.dataset):
            data_path = F"{str(path)}/file/data/"+args.dataset

            with open(data_path+'/input_label_pairs.json', "r", encoding="utf-8") as f:
                data = json.load(f)

            with open(data_path+'/conversation.txt', "w", encoding="utf-8") as f:
                for item in data:
                    input_text = item["input_text"]
                    label_text = item["label_text"]
                    f.write("Prompt: {}\n".format(input_text))
                    f.write("ChatBot: {}\n".format(label_text))
                    f.write("\n")
        else:
            print("Dataset not selected")


    # Change query_type of all pairs
    if args.option == "change_query_type":
        # python app\prepare_dataset.py -o change_query_type -d MentalFAQ
        if(args.dataset == "MentalFAQ"):
            dataset_path = F"{str(path)}/file/data/MentalFAQ/query_answer_pairs.json"
        else:
            print("Dataset not selected")
        
        data = load_from_json(dataset_path)
        query_type = "faq"

        for item in data:
            item.update({"query_type": str(query_type)})

        dump_to_json(data, dataset_path, sort_keys=False)

    # Append data to dataset
    if args.option == "append_data":
        # python app\prepare_dataset.py -o append_data -d MentalFAQ
        if(args.dataset == "MentalFAQ"):
            dataset_path = F"{str(path)}/file/data/MentalFAQ/query_answer_pairs.json"
            data_filepath = F"{str(path)}/file/test/MentalFAQ/MentalFAQ_test.json"
        else:
            print("Dataset not selected")
    
        dataset = load_from_json(dataset_path)
        data = load_from_json(data_filepath)

        query_type = "user_query"
        for item in data:
            item["query_type"] = str(query_type)

        dump_to_json(dataset+data, dataset_path, sort_keys=False)

    
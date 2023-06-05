import argparse
from pathlib import Path
from shared import *
from transformation import create_dataset

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--option", type=str, help="select an option", required=True)
    parser.add_argument("-d", "--dataset", type=str, help="select a dataset name", default="MentalFAQ")

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

    
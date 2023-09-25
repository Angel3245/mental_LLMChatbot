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
from pathlib import Path
from shared import *
from transformation import create_dataset

if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog='Prepare dataset',
                    description='Create a dataset for model training',
                    epilog='Jose Angel Perez Garrido - 2023')
    parser.add_argument("-o", "--option", type=str, help="select an option: parsing_text_generation -> Parse a dataset for text generation", default="parsing_text_generation")
    parser.add_argument("-d", "--dataset", type=str, help="select a dataset name (default: MentalKnowledge)", default="MentalKnowledge")

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

            # Create dataset applying transformation on input data 
            dataset_processed = create_dataset(file_path, posts_path, comments_path)

            # Dump data to json file
            make_dirs(output_path)
            dump_to_json(dataset_processed, output_path+'/input_label_pairs.json', sort_keys=False)

            print("Dataset",args.dataset, "created.")
        else:
            raise ValueError("Dataset not selected")

    
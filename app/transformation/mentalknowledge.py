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

from parsers import MentalFAQ_Parser, Reddit_Parser

def create_dataset(file_path, posts_path, comments_path):
    """
    The function creates a dataset by combining data from separate files.
    
    :param file_path: The path to the file containing the Mental_Health_FAQ.csv.
    :param posts_path: The path to the file containing the Reddit posts data
    :param comments_path: The path to the file containing the Reddit comments data
    """
    # create instance of MentalFAQ_Parser and generate query_answer_pairs
    mentalfaq_parser = MentalFAQ_Parser(file_path)
    print("Loaded",len(mentalfaq_parser.faq_pairs),"rows from Mental_Health_FAQ.csv")

    # create instance of Reddit_Parser and generate query_answer_pairs
    reddit_parser = Reddit_Parser(posts_path, comments_path)
    print("Loaded",len(reddit_parser.faq_pairs),"rows from Reddit posts and comments")

    # get query_answer_pairs
    query_answer_pairs = mentalfaq_parser.faq_pairs + reddit_parser.faq_pairs
    print("Total rows loaded:",len(query_answer_pairs))
    
    return query_answer_pairs
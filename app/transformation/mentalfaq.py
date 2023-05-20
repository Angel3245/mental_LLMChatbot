from parsers.mentalfaq import MentalFAQ_Parser
from preprocessing import *
import pandas as pd

def parse_mentalfaq(file_path):
    # read data as pandas DataFrame
    df = pd.read_csv(file_path)

    # rename colnames Questions: input_text, Answers: input_text
    df = df.rename(columns={'Questions': 'input_text','Answers':'label_text'})

    # drop null values
    df.dropna(inplace=True)

    # clean text
    df['input_text']=df['input_text'].map(lambda s:clean(s))

    # create instance of MentalFAQ_Parser and generate query_answer_pairs
    mentalfaq_parser = MentalFAQ_Parser()
    mentalfaq_parser.extract_data(df)

    # get faq_pairs
    return mentalfaq_parser.faq_pairs
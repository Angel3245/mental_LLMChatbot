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

import pandas as pd
from clean_data import clean

class MentalFAQ_Parser(object):
    """ Class for parsing & extracting data from Mental_Health_FAQ.csv """

    def __init__(self, file_path):
        # read data as pandas DataFrame
        df = pd.read_csv(file_path)

        df = self.prepare_data(df)

        self.faq_pairs = []
        self.num_faq_pairs = 0

        self.extract_data(df)
        
    def prepare_data(self, df):
        # rename colnames Questions: input_text, Answers: label_text
        df = df.rename(columns={'Questions': 'input_text','Answers':'label_text'})

        # drop null values
        df.dropna(inplace=True)

        # clean text
        df['input_text']=df['input_text'].map(lambda s:clean(s))

        return df
    
    def extract_pairs(self, df, query_type):
        """ Extract qa pairs from DataFrame for a given query_type
    
        :param df: input DataFrame
        :param query_type: faq
        :return: qa pairs
        """
        qa_pairs = []
        if query_type == "faq":
            # select question, answer columns
            df = df[['input_text', 'label_text']]

            for _, row in df.iterrows():
                data = dict()
                #data["query_type"] = "faq"
                data["prompt"] = row["input_text"]
                data["completion"] = row["label_text"]
                qa_pairs.append(data)

        else:
            raise ValueError('error, no query_type found for {}'.format(query_type))

        # remove duplicates
        pairs = []
        for pair in qa_pairs:
            if pair not in pairs:
                pairs.append(pair)

        return pairs
            
    def extract_data(self, df):
        """ Extract data from DataFrame
        
        :param df: Pandas DataFrame
        """
        # extract faq_pairs
        faq_pairs = self.extract_pairs(df, query_type='faq')
        
        self.faq_pairs = faq_pairs
        self.num_faq_pairs = len(faq_pairs)

        
import pandas as pd
from clean_dataset import clean

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
        # rename colnames Questions: question, Answers: answer
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

        
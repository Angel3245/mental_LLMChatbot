import pandas as pd
from clean_data import clean

class Reddit_Parser(object):
    """ Class for parsing & extracting data from Reddit_comments.csv & Reddit_posts.csv """

    def __init__(self, posts_path, comments_path):
        # read data as pandas DataFrame
        posts_df = pd.read_csv(posts_path)
        comments_df = pd.read_csv(comments_path)

        df = self.prepare_data(posts_df, comments_df)

        self.faq_pairs = []
        self.num_faq_pairs = 0
        self.support_pairs = []
        self.num_support_pairs = 0
        self.input_label_pairs = []
        self.num_input_label_pairs = 0

        self.extract_data(df)
        
    def filter_irrelevant_posts(self, posts_df):
        # filter by flair
        flairs = ["DAE Questions", "Question", ":snoo_thoughtful: help? :snoo_biblethump:",
                ":orly: Help please!", "DAE?",
                
                    "Needs A Hug/Support", "Need Support", ":snoo_hug: send support :snoo_sad:", "Advice",
                    "Advice Needed", "Support", "Seeking Support", "PROVIDING SUPPORT", "REQUESTING SUPPORT",
                    "PROVIDING ADVICE", "REQUESTING ADVICE"]
        posts_df = posts_df.apply(lambda row: row[posts_df['link_flair_text'].isin(flairs)])

        return posts_df

    def filter_irrelevant_comments(self, comments_df):
        # create a new comments_length column that contains the number of words per comment:
        comments_df["comment_length"] = comments_df.apply(
            lambda x: len(x["body"].split()), axis=1
        )

        # remove comments with low scores
        comments_df = comments_df[comments_df['score'] > 3]

        return comments_df
        
    def prepare_data(self, posts_df, comments_df):
        # drop null values
        posts_df.dropna(inplace=True)
        comments_df.dropna(inplace=True)

        #filter posts columns
        columns = posts_df.columns
        columns_to_keep = ["title", "body", "score", "name", "link_flair_text"]
        columns_to_remove = set(columns_to_keep).symmetric_difference(columns)
        posts_df = posts_df.drop(columns_to_remove, axis=1)
        # rename colnames score: post_score, body: post_body 
        posts_df = posts_df.rename(columns={'score': 'post_score','body': 'post_body'})

        # filter comments columns
        columns = comments_df.columns
        columns_to_keep = ["body", "score", "parent_id"]
        columns_to_remove = set(columns_to_keep).symmetric_difference(columns)
        comments_df = comments_df.drop(columns_to_remove, axis=1)
        
        # filter irrelevant data
        posts_df = self.filter_irrelevant_posts(posts_df)
        comments_df = self.filter_irrelevant_comments(comments_df)

        # join both dataframes
        df = posts_df.merge(comments_df,left_on='name', right_on='parent_id')

        # create prompt column Using + operator to combine title and body columns
        # Combine prompt
        df["prompt"] = df['title'].astype(str) + " " + df["post_body"]

        # rename colname body to answer
        df = df.rename(columns={'body': 'answer'})

        # clean text
        df['prompt']=df['prompt'].map(lambda s:clean(s)) 
        df['answer']=df['answer'].map(lambda s:clean(s))

        return df
    
    def extract_pairs(self, df, query_type):
        """ Extract qa pairs from DataFrame for a given query_type
    
        :param df: input DataFrame
        :param query_type: faq or support
        :return: qa pairs
        """
        qa_pairs = []
        if query_type == "faq":
            flairs = ["DAE Questions", "Question", ":snoo_thoughtful: help? :snoo_biblethump:",
                ":orly: Help please!", "DAE?"]

            # select prompt, answer columns
            df = df[['prompt', 'answer','link_flair_text']]

            for _, row in df.iterrows():
                # Check is a valid flair
                if row['link_flair_text'] in flairs:
                    data = dict()
                    #data["query_type"] = "faq"
                    data["prompt"] = row["prompt"]
                    data["completion"] = row["answer"]
                    qa_pairs.append(data)

        elif query_type == "support":
            flairs = ["Needs A Hug/Support", "Need Support", ":snoo_hug: send support :snoo_sad:", "Advice",
                    "Advice Needed", "Support", "Seeking Support", "PROVIDING SUPPORT", "REQUESTING SUPPORT",
                    "PROVIDING ADVICE", "REQUESTING ADVICE"]
            
            # select prompt, answer columns
            df = df[['prompt', 'answer','link_flair_text']]

            for _, row in df.iterrows():
                # Check is a valid flair
                if row['link_flair_text'] in flairs:
                    data = dict()
                    #data["query_type"] = "support"
                    data["prompt"] = row["prompt"]
                    data["completion"] = row["answer"]
                    qa_pairs.append(data)
        else:
            raise ValueError('error, no query_type found for {}'.format(query_type))

        # remove duplicates
        pairs = []
        for pair in qa_pairs:
            if pair not in pairs:
                pairs.append(pair)

        return pairs

    def get_input_label_pairs(self, faq_pairs, support_pairs):
        """ Generate query answer pair list using faq, support pairs 
        
        :param faq_pairs: faq pairs
        :param support_pairs: support pairs
        :return: query answer pairs
        """
        input_label_pairs = faq_pairs + support_pairs
        
        return input_label_pairs
          
    def extract_data(self, df):
        """ Extract data from DataFrame
        
        :param df: Pandas DataFrame
        """
        # extract faq_pairs, support_pairs
        faq_pairs = self.extract_pairs(df, query_type='faq')
        support_pairs = self.extract_pairs(df, query_type='support')
        input_label_pairs = self.get_input_label_pairs(faq_pairs, support_pairs)
        
        self.faq_pairs = faq_pairs
        self.num_faq_pairs = len(faq_pairs)
        self.support_pairs = support_pairs
        self.num_support_pairs = len(support_pairs)
        self.input_label_pairs = input_label_pairs
        self.num_input_label_pairs = len(input_label_pairs)

        
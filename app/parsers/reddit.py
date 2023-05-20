class Reddit_Parser(object):
    """ Class for parsing & extracting data from Reddit_comments.csv & Reddit_posts.csv """

    def __init__(self):
        self.faq_pairs = []
        self.num_faq_pairs = 0
        self.support_pairs = []
        self.num_support_pairs = 0
        self.input_label_pairs = []
        self.num_input_label_pairs = 0
        
    def extract_pairs(self, df, query_type):
        """ Extract qa pairs from DataFrame for a given query_type
    
        :param df: input DataFrame
        :param query_type: faq or user_query
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
                    data["query_type"] = "faq"
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
                    data["query_type"] = "support"
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
        """ Generate query answer pair list using faq, user query pairs 
        
        :param faq_pairs: faq pairs
        :param user_query_pairs: user query pairs
        :return: query answer pairs
        """
        query_answer_pairs = faq_pairs + support_pairs
        
        return query_answer_pairs
          
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

        
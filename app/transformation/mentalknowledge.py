from parsers import MentalFAQ_Parser, Reddit_Parser

def create_dataset(file_path, posts_path, comments_path):
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
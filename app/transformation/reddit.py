from parsers.reddit import Reddit_Parser
from preprocessing import *
import pandas as pd

def filter_irrelevant_posts(posts_df):
    # filter by flair
    flairs = ["DAE Questions", "Question", ":snoo_thoughtful: help? :snoo_biblethump:",
              ":orly: Help please!", "DAE?",
              
                "Needs A Hug/Support", "Need Support", ":snoo_hug: send support :snoo_sad:", "Advice",
                "Advice Needed", "Support", "Seeking Support", "PROVIDING SUPPORT", "REQUESTING SUPPORT",
                "PROVIDING ADVICE", "REQUESTING ADVICE"]
    posts_df = posts_df.apply(lambda row: row[posts_df['link_flair_text'].isin(flairs)])

    # create a new title_length column that contains the number of words per title:
    posts_df["title_length"] = posts_df.apply(
        lambda x: len(x["title"].split()), axis=1
    )
    # remove short posts
    posts_df = posts_df[posts_df['title_length'] > 4]

    return posts_df

def filter_irrelevant_comments(comments_df):
    # create a new comments_length column that contains the number of words per comment:
    comments_df["comment_length"] = comments_df.apply(
        lambda x: len(x["body"].split()), axis=1
    )

    # filter out short comments, which typically include things “Thanks!” that are not relevant for our search engine.
    comments_df = comments_df[comments_df["comment_length"] > 10]

    # remove answers with low scores
    comments_df = comments_df[comments_df['score'] > 3]

    # get answer with best score
    comments_df = comments_df.sort_values('score', ascending=False).drop_duplicates(['parent_id'])

    return comments_df

def parse_redditposts_textgeneration(posts_path, comments_path):
    # read data as pandas DataFrame
    posts_df = pd.read_csv(posts_path)
    comments_df = pd.read_csv(comments_path)

    # drop null values
    posts_df.dropna(inplace=True)
    comments_df.dropna(inplace=True)

    #filter posts columns
    columns = posts_df.columns
    columns_to_keep = ["title", "score", "name", "link_flair_text"]
    columns_to_remove = set(columns_to_keep).symmetric_difference(columns)
    posts_df = posts_df.drop(columns_to_remove, axis=1)
    # rename colnames score: post_score
    posts_df = posts_df.rename(columns={'score': 'post_score'})

    # filter comments columns
    columns = comments_df.columns
    columns_to_keep = ["body", "score", "parent_id"]
    columns_to_remove = set(columns_to_keep).symmetric_difference(columns)
    comments_df = comments_df.drop(columns_to_remove, axis=1)
    
    # filter irrelevant data
    posts_df = filter_irrelevant_posts(posts_df)
    comments_df = filter_irrelevant_comments(comments_df)

    # join both dataframes
    df = posts_df.merge(comments_df,left_on='name', right_on='parent_id')

    # rename title column to input_text
    df = df.rename(columns={'title': 'input_text'})

    # rename colname body: label_text
    df = df.rename(columns={'body': 'label_text'})

    # clean text
    df['input_text']=df['input_text'].map(lambda s:preprocess(s)) 
    df['label_text']=df['label_text'].map(lambda s:clean(s))

    # create instance of Reddit_Parser and generate input_label_pairs
    reddit_parser = Reddit_Parser()
    reddit_parser.extract_data(df)

    # get input_label_pairs
    return reddit_parser.input_label_pairs
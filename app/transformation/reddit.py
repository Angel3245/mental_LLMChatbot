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

    return posts_df

def filter_irrelevant_comments(comments_df):
    # remove comments with low scores
    comments_df = comments_df[comments_df['score'] > 3]

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
    posts_df = filter_irrelevant_posts(posts_df)
    comments_df = filter_irrelevant_comments(comments_df)

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

    # create instance of Reddit_Parser and generate input_label_pairs
    reddit_parser = Reddit_Parser()
    reddit_parser.extract_data(df)

    # get input_label_pairs
    return reddit_parser.input_label_pairs
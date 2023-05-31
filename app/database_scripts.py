import argparse
from pathlib import Path
from transformation import *
from database.connect import database_connect
from model import *
import csv
import pandas as pd

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--option", type=str, help="select an option", required=True)
    parser.add_argument("-d", "--dataset", type=str, help="select a dataset", default="MentalFAQ")
    parser.add_argument("-db", "--database", type=str, help="select a database")
    parser.add_argument("-f", "--file", type=str, help="select a file")
    args = parser.parse_args()

    path = Path.cwd()

    # Create csv from data in DB
    if args.option == "create_csv_from_DB":
        # python app\database_scripts.py -o create_csv_from_DB -db <<database_name>>
        # Example: python app\database_scripts.py -o create_csv_reddit -db reddit
        session = database_connect(args.database)

        outfile = open(F"{str(path)}/file/datasets/Reddit_posts.csv", 'w', encoding='utf-8')
        outcsv = csv.writer(outfile)
        records = session.query(Post).all()

        outcsv.writerow(['id', 'name', 'user_id', 'subreddit_id', 'permalink', 'body', 'body_html', 'title', 'created_utc', 'downs',
                    'no_follow', 'score', 'send_replies', 'stickied', 'ups', 'link_flair_text', 'link_flair_type'])
        for item in records: 
            outcsv.writerow([item.id, item.name, item.user_id, item.subreddit_id, item.permalink, item.body, item.body_html, 
                             item.title, item.created_utc, item.downs, item.no_follow, item.score, item.send_replies,
                             item.stickied, item.ups, item.link_flair_text, item.link_flair_type])

        outfile.close()

        outfile = open(F"{str(path)}/file/datasets/Reddit_comments.csv", 'w', encoding='utf-8')
        outcsv = csv.writer(outfile)
        records = session.query(Comment).all()
        
        outcsv.writerow(['id', 'name', 'user_id', 'subreddit_id', 'body', 'body_html', 'created_utc', 'downs',
                    'no_follow', 'score', 'send_replies', 'stickied', 'ups', 'permalink', 'parent_id'])
        for item in records: 
            outcsv.writerow([item.id, item.name, item.user_id, item.subreddit_id, item.body, item.body_html, 
                             item.created_utc, item.downs, item.no_follow, item.score, item.send_replies,
                             item.stickied, item.ups, item.permalink, item.parent_id])

        outfile.close()

    print("PROGRAM FINISHED")
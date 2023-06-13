from database.connect import database_connect
import argparse, csv
from extraction.reddit import extract_data
from extraction.refresh import refresh_token
from pathlib import Path
from model import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Reddit scripts',
                    description='Get and convert data from Reddit',
                    epilog='Jose Angel Perez Garrido - 2023')

    parser.add_argument("-o", "--option", type=str, help="select an option: extraction_search_by_flair -> get data from Reddit searching by flair; refresh_token -> get refresh token; create_csv_from_DB -> create csv datasets from Reddit data in DB")
    parser.add_argument("-s", "--subreddit", type=str, help="name of subreddit", required=True)
    parser.add_argument("-d", "--database", type=str, help="name of database", required=True)
    parser.add_argument("-f", "--flairs", type=str, help="flairs to search separated by commas", required=True)

    args = parser.parse_args()
    path = Path.cwd()

    # Search by flair in Reddit subreddits
    if args.option == "extraction_search_by_flair":
        # python app\reddit_scripts.py -o extraction_search_by_flair -s Anxiety -d reddit
        flairs = [str(item) for item in args.flair.split(',')]

        session = database_connect(args.database)
        extract_data(session, args.subreddit, "search_by_flair", flairs)

    # Create refresh token for Reddit extraction
    if args.option == "refresh_token":
        # python app\reddit_scripts.py -o refresh_token
        refresh_token()

    # Create csv from data in Reddit DB
    if args.option == "create_csv_from_DB":
        # python app\reddit_scripts.py -o create_csv_reddit -d reddit
        session = database_connect(args.database)

        # POSTS
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

        # COMMENTS
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
from database.connect import database_connect
import argparse
from extraction.reddit import extract_data
from refresh.refresh import refresh_token


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-o", "--option", type=str, help="select an option")
    parser.add_argument("-s", "--subreddit", type=str, help="name of subreddit", required=True)
    parser.add_argument("-d", "--database", type=str, help="name of database", required=True)
    parser.add_argument("-f", "--flairs", type=str, help="flairs to search separated by commas", required=True)

    args = parser.parse_args()

    if args.option == "extraction_search_by_flair":
        # python app\reddit_scripts.py -o extraction_search_by_flair -s Anxiety -d reddit
        flairs = [str(item) for item in args.flair.split(',')]

        session = database_connect(args.database)
        extract_data(session, args.subreddit, "search_by_flair", flairs)

    if args.option == "refresh_token":
        # python app\reddit_scripts.py -o refresh_token
        refresh_token()
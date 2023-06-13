from datetime import datetime
import json
import praw
from pathlib import Path
import time
from model import Post, User, Comment, Subreddit

def extract_data(session, subreddit_name: str, type_query: str, query = ""):
    """ Extract data from a subreddit for a given query
    
        :param session: input SQLAlchemy database session to store data
        :param subreddit_name: input subreddit name
        :param type_query: search type (search_by_flair)
        :param query: search condition (flairs)
    """

    path = Path.cwd()

    # Load Reddit credentials
    with open(f"{str(path)}/file/reddit/client_secrets.json") as json_file:
        data = json.load(json_file)

    reddit = praw.Reddit(client_id= data['client_id'],
                        client_secret= data['client_secret'],
                        user_agent= data['user_agent'],
                         )

    print("Start extraction post.....")

    sub = reddit.subreddit(subreddit_name)

    subreddit_db = session.query(Subreddit).get(sub.id)

    if not subreddit_db:
        subreddit = Subreddit()
        subreddit_created_utc = datetime.utcfromtimestamp(sub.created_utc).strftime('%Y-%m-%d %H:%M:%S')
        subreddit.load_data(sub.id, sub.name, sub.display_name, sub.display_name_prefixed, sub.title, sub.url,
                            subreddit_created_utc, sub.description, sub.description_html, sub.public_description,
                            sub.public_description_html, sub.submit_text, sub.submit_text_html, sub.submit_text_label,
                            sub.accept_followers, sub.accounts_active, sub.active_user_count, sub.can_assign_link_flair,
                            sub.can_assign_user_flair, sub.lang, sub.subscribers, sub.wiki_enabled)

        session.add(subreddit)
        session.commit()

    print(f"Extraction subreddit {subreddit_name}")

    # Search by flair
    if type_query == "search_by_flair":
        for q in query:
            print(f"Extraction flair: {q}")
            for i, submission in enumerate(reddit.subreddit(subreddit_name).search(f"flair:{q}", limit=None)):
                try:
                    post_db = session.query(Post).get(submission.id)

                    if not post_db:

                        if submission.author:
                            if hasattr(submission.author, 'id'):
                                _comprobe_user(session, submission.author.id, submission.author)

                                # ------- POSTS ----------------------
                                post = Post()
                                post_created_utc = datetime.utcfromtimestamp(submission.created_utc).strftime('%Y-%m-%d %H:%M:%S')
                                post.load_data(submission.id, submission.name, submission.author.id, submission.subreddit_id,
                                           submission.permalink, submission.selftext, submission.selftext_html, submission.title,
                                           post_created_utc, submission.downs, submission.no_follow, submission.score,
                                           submission.send_replies, submission.stickied, submission.ups, submission.link_flair_text,
                                           submission.link_flair_type)

                                session.add(post)
                                session.commit()

                        # ------- COMMENTS ----------------------
                        for j, replies in enumerate(submission.comments):
                            if replies.author:

                                comment_db = session.query(Comment).get(replies.id)
                                if not comment_db:
                                    comment = Comment()
                                    if hasattr(replies.author, 'id'):
                                        _comprobe_user(session, replies.author.id, replies.author)
                                        replies_created_utc = datetime.utcfromtimestamp(replies.created_utc).strftime('%Y-%m-%d %H:%M:%S')

                                        comment.load_data(replies.id, replies.name, replies.author.id, replies.subreddit_id, replies.body,
                                                          replies.body_html, replies_created_utc, replies.downs, replies.no_follow, replies.score,
                                                          replies.send_replies, replies.stickied, replies.ups, replies.permalink, replies.parent_id)

                                        session.add(comment)
                                        session.commit()
                except Exception as e:
                    print("sleep.....")
                    print(e)
                    time.sleep(10)  # Sleeping For 2 seconds to resolve the server overload error

                if i % 10 == 0 and i != 0:
                    print(f"Insert data: {i}")



def _comprobe_user(session, author_id, author):

    user_db = session.query(User).get(author_id)

    if not user_db:
        user = User()

        user_created_utc = datetime.utcfromtimestamp(author.created_utc).strftime('%Y-%m-%d %H:%M:%S')
        user.load_data(author.id, author.name, user_created_utc, author.subreddit, author.awardee_karma,
                       author.awarder_karma, author.comment_karma, author.has_subscribed,
                       author.has_verified_email, author.hide_from_robots, author.icon_img, author.is_employee,
                       author.is_gold, author.is_mod, author.link_karma, author.pref_show_snoovatar,
                       author.snoovatar_img, author.total_karma, author.verified)

        session.add(user)
        session.commit()

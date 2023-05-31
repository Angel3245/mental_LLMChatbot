from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class Subreddit(Base):
    """
    Representa un subreddit.
    """

    __tablename__ = 'Subreddit'

    id = Column(String(50), primary_key=True, nullable=False, unique=True)
    name = Column(String(50), nullable=False, unique=True)
    display_name = Column(String(50), nullable=False)
    display_name_prefixed = Column(String(50), nullable=False)
    title = Column(String(Text), nullable=False)
    url = Column(String(Text), nullable=False)
    created_utc = Column(DateTime, nullable=False)
    description = Column(Text)
    description_html = Column(Text)
    public_description = Column(Text)
    public_description_html = Column(Text)
    submit_text = Column(Text)
    submit_text_html = Column(Text)
    submit_text_label = Column(String(100))
    accept_followers = Column(Boolean)
    accounts_active = Column(Integer)
    active_user_count = Column(Integer)
    can_assign_link_flair = Column(Boolean)
    can_assign_user_flair = Column(Boolean)
    lang = Column(String(5))
    subscribers = Column(Integer)
    wiki_enabled = Column(Boolean)

    def load_data(self, id, name, display_name, display_name_prefixed, title, url, created_utc, description,
                  description_html, public_description, public_description_html, submit_text, submit_text_html,
                  submit_text_label, accept_followers, accounts_active, active_user_count, can_assign_link_flair,
                  can_assign_user_flair, lang, subscribers, wiki_enabled):

        self.id = id
        self.name = name
        self.display_name = display_name
        self.display_name_prefixed = display_name_prefixed
        self.title = title
        self.url = url
        self.created_utc = created_utc
        self.description = description
        self.description_html = description_html
        self.public_description = public_description
        self.public_description_html = public_description_html
        self.submit_text = submit_text
        self.submit_text_html = submit_text_html
        self.submit_text_label = submit_text_label
        self.accept_followers = accept_followers
        self.accounts_active = accounts_active
        self.active_user_count = active_user_count
        self.can_assign_link_flair = can_assign_link_flair
        self.can_assign_user_flair = can_assign_user_flair
        self.lang = lang
        self.subscribers = subscribers
        self.wiki_enabled = wiki_enabled





from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class User(Base):
    """ ORM entity for Reddit User
    """

    __tablename__ = 'User'

    id = Column(String(50), nullable=False, primary_key=True, unique=True)
    name = Column(String(50), nullable=False)
    created_utc = Column(DateTime, nullable=False)
    subreddit = Column(String(50))
    awardee_karma = Column(Integer)
    awarder_karma = Column(Integer)
    comment_karma = Column(Integer)
    has_subscribed = Column(Boolean)
    has_verified_email = Column(Boolean)
    hide_from_robots = Column(Boolean)
    icon_img = Column(Text)
    is_employee = Column(Boolean)
    is_gold = Column(Boolean)
    is_mod = Column(Boolean)
    link_karma = Column(Integer)
    pref_show_snoovatar = Column(Boolean)
    snoovatar_img = Column(Text)
    total_karma = Column(Integer)
    verified = Column(Boolean)

    def load_data(self, id, name, created_utc, subreddit, awardee_karma, awarder_karma, comment_karma, has_subscribed,
                    has_verified_email, hide_from_robots, icon_img, is_employee, is_gold, is_mod, link_karma,
                    pref_show_snoovatar, snoovatar_img, total_karma, verified):

        self.id = id
        self.name = name
        self.created_utc = created_utc
        self.subreddit = subreddit
        self.awardee_karma = awardee_karma
        self.awarder_karma = awarder_karma
        self.comment_karma = comment_karma
        self.has_subscribed = has_subscribed
        self.has_verified_email = has_verified_email
        self.hide_from_robots = hide_from_robots
        self.icon_img = icon_img
        self.is_employee = is_employee
        self.is_gold = is_gold
        self.is_mod = is_mod
        self.link_karma = link_karma
        self.pref_show_snoovatar = pref_show_snoovatar
        self.snoovatar_img = snoovatar_img
        self.total_karma = total_karma
        self.verified = verified

# Copyright (C) 2023  Jose Ángel Pérez Garrido
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class Post(Base):
    """ ORM entity for Reddit Post
    """

    __tablename__ = 'Post'

    id = Column(String(50), primary_key=True, nullable=False)
    name = Column(String(50), nullable=False, unique=True)
    user_id = Column(String(50), nullable=False)
    subreddit_id = Column(String(50), nullable=False)
    permalink = Column(Text)
    body = Column(Text)
    body_html = Column(Text)
    title = Column(Text)
    created_utc = Column(DateTime)
    downs = Column(Integer)
    no_follow = Column(Boolean)
    score = Column(Integer)
    send_replies = Column(Boolean)
    stickied = Column(Boolean)
    ups = Column(Integer)
    link_flair_text = Column(String(50))
    link_flair_type = Column(String(50))

    def load_data(self, id, name, user_id, subreddit_id, permalink, body, body_html, title, created_utc, downs,
                    no_follow, score, send_replies, stickied, ups, link_flair_text, link_flair_type):

        self.id = id
        self.name = name
        self.user_id = user_id
        self.subreddit_id = subreddit_id
        self.permalink = permalink
        self.body = body
        self.body_html = body_html
        self.title = title
        self.created_utc = created_utc
        self.downs = downs
        self.no_follow = no_follow
        self.score = score
        self.send_replies = send_replies
        self.stickied = stickied
        self.ups = ups
        self.link_flair_text = link_flair_text
        self.link_flair_type = link_flair_type
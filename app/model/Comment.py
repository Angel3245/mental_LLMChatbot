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

class Comment(Base):
    """ ORM entity for Reddit Comment
    """

    __tablename__ = 'Comment'

    id = Column(String(50), primary_key=True, nullable=False)
    name = Column(String(50), nullable=False, unique=True)
    user_id = Column(String(50), nullable=False)
    subreddit_id = Column(String(50), nullable=False)
    body = Column(Text)
    body_html = Column(Text)
    created_utc = Column(DateTime)
    downs = Column(Integer)
    no_follow = Column(Boolean)
    score = Column(Integer)
    send_replies = Column(Boolean)
    stickied = Column(Boolean)
    ups = Column(Integer)
    permalink = Column(Text)
    parent_id = Column(String(50))


    def load_data(self, id, name, user_id, subreddit_id, body, body_html, created_utc, downs, no_follow, score,
                    send_replies, stickied, ups, permalink, parent_id):

        self.id = id
        self.name = name
        self.user_id = user_id
        self.subreddit_id = subreddit_id
        self.body = body
        self.body_html = body_html
        self.created_utc = created_utc
        self.downs = downs
        self.no_follow = no_follow
        self.score = score
        self.send_replies = send_replies
        self.stickied = stickied
        self.ups = ups
        self.permalink = permalink
        self.parent_id = parent_id
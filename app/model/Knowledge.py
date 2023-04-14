from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, String

Base = declarative_base()
class Knowledge(Base):
    __tablename__ = 'KNOWLEDGE'
    question = Column(String, primary_key=True)
    answer = Column(String)
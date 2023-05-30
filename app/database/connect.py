from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine

def database_connect(target: str = "reddit"):

    if(target == "reddit"):
        engine = create_engine(
            f"mysql+pymysql://redditUser:redditPass@localhost:3306/{target}?charset=utf8mb4"
        )
        Session = sessionmaker(bind=engine, autoflush=False)

    return Session()
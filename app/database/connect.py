from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine

def database_connect(target: str = "reddit"):
    """ Create a connection to a MySQL Database
    
        :param target: database name
        :return: SQLAlchemy database session
    """
    if(target == "reddit"):
        engine = create_engine(
            f"mysql+pymysql://redditUser:redditPass@localhost:3306/{target}?charset=utf8mb4"
        )
        Session = sessionmaker(bind=engine, autoflush=False)

    return Session()
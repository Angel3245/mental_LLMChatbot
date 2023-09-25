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
from flask_marshmallow import Marshmallow
from flask_sqlalchemy import SQLAlchemy
import config

ma = Marshmallow()
db = SQLAlchemy()

from sqlalchemy import MetaData, Table, create_engine
from sqlalchemy.ext.declarative import declarative_base
Base = declarative_base()

engine = create_engine(config.SQLALCHEMY_DATABASE_URI, encoding='utf8', echo=True)
metadata = MetaData(bind=engine)

class Novel(Base):
    __table__ = Table('NOVEL', metadata, auto_increment=True, autoload=True)

class Contents(Base):
    __table__ = Table('CONTENTS', metadata, auto_increment=True, autoload=True)
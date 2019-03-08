# -*- coding: utf-8 -*-
import config

from sqlalchemy import MetaData, Table
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
Base = declarative_base()

# Model Example 
#from sqlalchemy import Column, Integer, String
#
#class User(Base):
#    __tablename__ = 'users'
#
#    id = Column(Integer, primary_key=True)
#    name = Column(String(length = 500))
#    fullname = Column(String(length = 500))
#    password = Column(String(length = 500))
#
#    def __init__(self, name, fullname, password):
#        self.name = name
#        self.fullname = fullname
#        self.password = password
#
#    def __repr__(self):
#        return "<User('%s', '%s', '%s')>" % (self.name, self.fullname, self.password)
    
# DB Model Auto Generate
engine = create_engine(config.SQLALCHEMY_DATABASE_URI, encoding='utf8', echo=False)
metadata = MetaData(bind=engine)

class Novel(Base):
    __table__ = Table('NOVEL', metadata, auto_increment=True, autoload=True)

class Contents(Base):
    __table__ = Table('CONTENTS', metadata, auto_increment=True, autoload=True)

    

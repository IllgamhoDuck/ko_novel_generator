# -*- coding: utf-8 -*-

import os

basedir = os.path.abspath(os.path.dirname(__file__))
SQLALCHEMY_ECHO = True
SQLALCHEMY_TRACK_MODIFICATIONS = True
SQLALCHEMY_DATABASE_URI = 'mysql://{USER_NAME}:{PASSWORD}@{DB_END_POINT}:{DB_PORT}/{DB_NAME}?charset=utf8'

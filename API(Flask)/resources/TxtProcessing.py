# -*- coding: utf-8 -*-
import sys
sys.path.append("..")

import config
from flask_restful import Resource
from flask import request
from gru_train.generate import generate_run

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker    
from resources.DbModels import Contents
    
class PutHumanTxt(Resource):
    def get(self):
        contentsId = request.args.get('contents_id')
        isFirst = request.args.get('is_first')
        get_txt_from_db(contentsId, isFirst)
    
    def post(self):
        contentsId = request.args.get('contents_id')
        isFirst = request.args.get('is_first')
        get_txt_from_db(contentsId, isFirst)
        get_txt_from_db(contentsId, isFirst)
    

    

def get_txt_from_db(contentsId, isFirst):
    # Create and engine and get the metadata
    engine = create_engine(config.SQLALCHEMY_DATABASE_URI, encoding='utf8', echo=False)
    Session = sessionmaker(bind=engine)
    Session.configure(bind=engine)
    conn = engine.connect()
    session = Session()
    
    contentsFromId = session.query(Contents).filter(Contents.ID == contentsId).scalar()
    resultTxt = generate_run(epoch=50, prime=contentsFromId.TEXT, resume=(False if isFirst == "Y" else True))
    print(resultTxt)
    
    aiTxt = Contents(NOVEL_ID=contentsFromId.NOVEL_ID
                     , USER_NAME='BlackOriBanana'
                     , CONTENTS_TYPE='AI'
                     , TEXT=resultTxt)
    print(aiTxt.TEXT)
    session.add(aiTxt)
    session.commit()
    
    
    # session close
    session.close()
    conn.close()
    engine.dispose()

# -*- coding: utf-8 -*-
import os
import sys
sys.path.append("..")
sys.path.append("../..")

import config
from flask_restful import Resource
from flask import request

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker    
from resources.DbModels import Contents

import pickle
import torch
from opt import opt as opt_
from generate import softmax, highest_epoch, word_discriminator, word_weight_discriminator, sample_model
from models import gru

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


def generate_run(epoch=None, prime=" ", length=100, resume=False):

#    assert epoch != None, "The epoch must be entered! --epoch [int]"
    
    # Directory
    # - Where is trainable data
    # - Where to save the model parameter
    # - Where to save the generated text
    # - Where to save the vocabulary
    data_dir = "data/"
    save_dir = "save/"
    gen_dir = "generate/"
    vocan_dir = "vocab/"

    # Choose the hyperparameter at here!
    ratio = 0.9
    num_layers = 2
    hidden_size = 1024
    embedding_size = 1024
    cuda = True if torch.cuda.is_available() else False
    batch_size = 1 # Change 1 because this is generating
    seq_len = 50
    num_epochs = 100
    save_every = 50
    print_every = 10
    valid_every = 20 # test the valid data when batch step is (int)
    grad_clip = 5.
    learning_rate = 0.001

    # Store every options to opt class data structure
    opt = opt_(data_dir=data_dir,
              save_dir=save_dir,
              gen_dir=gen_dir,
              ratio = ratio,
              num_layers=num_layers,
              hidden_size=hidden_size,
              embedding_size=embedding_size,
              cuda=cuda,
              batch_size=batch_size,
              seq_len = seq_len,
              num_epochs=num_epochs,
              save_every=save_every,
              print_every=print_every,
              valid_every=valid_every,
              grad_clip=grad_clip,
              learning_rate=learning_rate)
    
    if not epoch:
        epoch = highest_epoch(opt)
        
    # load the vocab data
    with open(os.path.join(vocan_dir, 'vocab.pkl'), 'rb') as f:
        vocab = pickle.load(f)

    # load the hidden data if resume is true
    if resume:
#        h = torch.load(os.path.join(opt.gen_dir, "hidden.pkl"), map_location='cpu')
        with open(os.path.join(opt.gen_dir, "hidden.pkl"), 'rb') as f:
#            h = torch.load(f, map_location=lambda storage, loc: storage)
            h = pickle.load(f)
    else:
        h = None
        
    # Store vocabulary to the option
    opt.vocab_size = vocab['vocab_size']
    opt.vocab_itoc = vocab['vocab_itoc']
    opt.vocab_ctoi = vocab['vocab_ctoi']
    
    # make model
    model = gru(opt)
    if opt.cuda:
        model = model.cuda()

    # load saved torch data
    save_path = os.path.join(opt.save_dir, "gru_{0}.pkl".format(epoch))

    # load saved model
    model.load_state_dict(torch.load(save_path, map_location='cpu'))

    result, h = sample_model(opt=opt,
                          model=model,
                          hidden=h,
                          prime_text=prime,
                          length=length)

    
    # store the hidden state
    with open(os.path.join(opt.gen_dir, "hidden.pkl"), 'wb') as f:
        pickle.dump(h, f)

    return result

import codecs
import os
import re
import math
import numpy as np

import torch
import torch.utils.data
from torch.autograd import Variable


class gru_dataset(torch.utils.data.Dataset):
    def __init__(self, text, opt):
        
        # Store the option
        self.opt = opt
        
        # Seperate text to sentence
        if "</s>" in text:
            sentences = text.split("</s>")
        else:
            raise Exception('Please delimit the data with </s>')
        
        # Split the data to train / validate - 9 : 1 ratio is default
        total_line = len(sentences)
        train_line = math.ceil(total_line * opt.ratio)
        
        train_data = sentences[:train_line]
        valid_data = sentences[train_line:]
        
        train_data = ' '.join(train_data)
        valid_data = ' '.join(valid_data)
        
        # Shorten the space width by regular expression
        train_data = re.sub('\s{2,}', repl=' ', string=train_data)
        valid_data = re.sub('\s{2,}', repl=' ', string=valid_data)
        
        window_size = opt.batch_size * opt.seq_len
        
        # Train data - store length of indexs and make it to char 
        train_data_num = len(train_data) // window_size
        self.train_num = train_data_num
        train_data = train_data[:window_size * train_data_num]
        train_data = [c for c in train_data]
        
        if len(train_data) < opt.batch_size * opt.seq_len:
            print("Train data is not enough long to use")
            opt.train = False
        
        # Valid data - store length of indexs and make it to char
        valid_data_num = len(valid_data) // window_size
        self.valid_num = valid_data_num
        valid_data = valid_data[:window_size * valid_data_num]
        valid_data = [c for c in valid_data]
        
        if len(valid_data) < opt.batch_size * opt.seq_len:
            print("valid data is not enough long to use")
            opt.valid = False
        
        # Preprocess the data - 0 represents UNK(Unknown)
        train_data = [opt.vocab_ctoi[c] if c in opt.vocab_ctoi else 0 for c in train_data]
        valid_data = [opt.vocab_ctoi[c] if c in opt.vocab_ctoi else 0 for c in valid_data]
        
        # Store the data to pytorch tensor
        train_data = torch.LongTensor(train_data).view(opt.batch_size, -1)
        valid_data = torch.LongTensor(valid_data).view(opt.batch_size, -1)
        
        # Split the data to x(input), y(target)
        train_data_x = train_data.clone()
        train_data_y = train_data.clone()
        if opt.train == True:
            train_data_y[:,:-1], train_data_y[:,-1] = train_data[:,1:], train_data[:,0]

        valid_data_x = valid_data.clone()
        valid_data_y = valid_data.clone()
        if opt.valid == True:
            valid_data_y[:,:-1], valid_data_y[:,-1] = valid_data[:,1:], valid_data[:,0]
        
        # Store the dataset to self.
        self.train_x = train_data_x
        self.train_y = train_data_y
        
        self.valid_x = valid_data_x
        self.valid_y = valid_data_y
        
        
    def __getitem__(self, id):
        if self.opt.mode == 'train':
            return (self.train_x[:,id*self.opt.seq_len:(id+1)*self.opt.seq_len],
                    self.train_y[:,id*self.opt.seq_len:(id+1)*self.opt.seq_len])
        elif self.opt.mode == 'valid':
            return (self.valid_x[:,id*self.opt.seq_len:(id+1)*self.opt.seq_len],
                    self.valid_y[:,id*self.opt.seq_len:(id+1)*self.opt.seq_len])
        else:
            raise Exception('Set the opt.mode to one of train / valid') 


    def __len__(self):
        if self.opt.mode == 'train':
            return self.train_num
        elif self.opt.mode == 'valid':
            return self.valid_num
        else:
            raise Exception('Set the opt.mode to one of train / valid') 

    
def load_data(opt):
    # load the novel
    with codecs.open(opt.novel_path, "r", encoding="utf-8") as f:
        novel_text = f.read()

    dataset = gru_dataset(text=novel_text, opt=opt)
    
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             num_workers=10,
                                             pin_memory=opt.cuda)
    return dataloader

import os
import pickle
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils
from torch.autograd import Variable

from model import gru
from data_loader import load_data
from opt import opt

# TODO - Make a file that will load the novel randomly or in order.

def train(opt):
    # Filling this title for temporary
    dataloader = load_data(opt)
    model = gru(opt)
    
    if opt.train == False:
        print("Finish the training session because the train data is too short")
        return
    
    # Multi-GPU Setting
    if opt.cuda:
        model.cuda()
        gpu_num = torch.cuda.device_count()
        print('GPU number is ', gpu_num)
        model = nn.DataParallel(model, device_ids=[i for i in range(gpu_num)], dim=1).cuda()
    
    if opt.resume:
        # load saved torch data
        if opt.resume_epoch:
            save_path = os.path.join(opt.save_dir, "gru_{0}.pkl".format(opt.resume_epoch))
        else:
            raise Exception("If your trying to transfer learn specify the epoch to [opt.resume_epoch]")
        # load saved model
        model.module.load_state_dict(torch.load(save_path))
                
    # Setup Training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    # Set model for training mode
    model.train()
    
    # TRAIN PROCESS
    opt.mode = 'train'
    total_batch_step_size = len(dataloader)
    
    # Valid switch - warn the valid is unable only once
    valid_announce = True
    
    for epoch in range(opt.num_epochs):
        # Init hidden variable
        h = model.module.init_hidden()
        h = Variable(h)
        if opt.cuda:
            h = h.cuda()
            
        for batch_i, (x, y) in enumerate(dataloader):
            
            # Pre-process inputs
            x = x.squeeze()
            y = y.squeeze()
            x = Variable(x)
            y = Variable(y)
            
            if opt.cuda:
                x = x.cuda()
                y = y.cuda()
            
            # Detach from the past
            h = h.detach()
            
            # why the parallel version needs this even the non parallel doesn't require this?
            # Supposing that the parallel gives back the [seq_len, batch_size, hidden_size]
            x = x.transpose(1, 0)
            y = y.transpose(1, 0)
            
            optimizer.zero_grad()
            output, h = model(x=x, h=h)
            
            loss = criterion(output.contiguous().view(-1, opt.vocab_size), y.contiguous().view(-1))
            loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
            optimizer.step()

            loss = loss.item()

            # Print status
            if (batch_i+1) % opt.print_every == 0 and opt.resume == None:
                print("Epoch={}/{}: Batch={}/{}: Loss={}".format(epoch+1, opt.num_epochs, batch_i+1,
                                                                 total_batch_step_size, loss))
            elif (batch_i+1) % opt.print_every == 0 and opt.resume != None:
                print("Epoch={}/{}: Batch={}/{}: Loss={}".format(epoch+1+opt.resume_epoch,
                                                                 opt.num_epochs+opt.resume_epoch, batch_i+1,
                                                                 total_batch_step_size, loss))
            # check validate
            if (batch_i+1) % opt.valid_every == 0:
                if opt.valid == True:
                    valid(opt, model, dataloader, criterion, optimizer)
                elif valid_announce == True:
                    print("Valid data is not enough to use")
                    # Turn of the announce so no need to talk about this anymore
                    valid_announce = False
        
        # Save the model parameter
        if opt.save_every and (epoch+1) % opt.save_every == 0 and opt.resume == None:
            save_path = os.path.join(opt.save_dir, "gru_{0}.pkl".format(epoch+1))
            print("Saving to {}".format(save_path))
            torch.save(model.module.state_dict(), save_path)
        elif opt.save_every and (epoch+1) % opt.save_every == 0 and opt.resume != None:
            save_path = os.path.join(opt.save_dir, "gru_{0}.pkl".format(epoch+1+opt.resume_epoch))
            print("Saving to {}".format(save_path))
            torch.save(model.module.state_dict(), save_path)


# Checking the validate
def valid(opt, model, dataloader, criterion, optimizer):
    
    # VALIDATE PROCESS
    opt.mode = 'valid'

    # Init hidden variable
    valid_h = model.module.init_hidden()
    valid_h = Variable(valid_h)
    if opt.cuda:
        valid_h = valid_h.cuda()

    valid_loss = 0 
    valid_iter_num = 0

    for x, y in dataloader:
        # Pre-process inputs
        x = x.squeeze()
        y = y.squeeze()
        x = Variable(x)
        y = Variable(y)

        if opt.cuda:
            x = x.cuda()
            y = y.cuda()
            
        # Detach from the past
        valid_h = valid_h.detach()
        
        # why the parallel version needs this even the non parallel doesn't require this?
        # Supposing that the parallel gives back the [seq_len, batch_size, hidden_size]
        x = x.transpose(1, 0)
        y = y.transpose(1, 0)

        optimizer.zero_grad()
        output, valid_h = model(x=x, h=valid_h)
        loss = criterion(output.contiguous().view(-1, opt.vocab_size), y.contiguous().view(-1))
        loss = loss.item()

        valid_loss += loss
        valid_iter_num += 1

    # Print status
    print("validation loss is : {}".format(valid_loss / valid_iter_num))

    # TRAIN PROCESS
    opt.mode = 'train'

            
if __name__ == "__main__":
    # Directory
    # - Where is trainable data
    # - Where to save the model parameter
    # - Where to save the generated text
    data_dir = "data/"
    save_dir = "save/"
    gen_dir = "generate/"

    # Choose the hyperparameter at here!
    ratio = 0.9
    num_layers = 2
    hidden_size = 1024
    embedding_size = 1024
    cuda = True if torch.cuda.is_available() else False
    batch_size = 8
    seq_len = 200
    num_epochs = 3
    save_every = 1
    print_every = 50
    valid_every = 50 # test the valid data when batch step is (int)
    grad_clip = 5.
    learning_rate = 0.001

    # Store every options to opt class data structure
    opt = opt(data_dir=data_dir,
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
    
    # load the vocab data
    with open('vocab/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
        
    # Store vocabulary to the option
    opt.vocab_size = vocab['vocab_size']
    opt.vocab_itoc = vocab['vocab_itoc']
    opt.vocab_ctoi = vocab['vocab_ctoi']
    
    # Specify the novel path
    opt.novel_path = "data/ov.txt"
    
    # Resume
    # opt.resume = True
    # opt.resume_epoch = 2
    
    opt.train = True
    opt.valid = True

    # Start training
    train(opt)
    

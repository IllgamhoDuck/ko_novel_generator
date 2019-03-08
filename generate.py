import os
import argparse

import numpy as np
import pickle

from model import gru
from data_loader import gru_dataset
from opt import opt

import torch
import torch.nn as nn
from torch.autograd import Variable

def softmax(output):
    theta = 1
    output /= np.exp(output * theta)
    output /= np.sum(output)
    return output

def highest_epoch(opt):
    epoch_list = os.listdir(opt.save_dir)
    epoch_list = [int(epoch[4:-4]) for epoch in epoch_list]
    highest_epoch = sorted(epoch_list)[-1]
    return highest_epoch

def word_discriminator(output):
    return output[-1,:,:].squeeze().max(dim=0)[1].item()

def word_weight_discriminator(output):
    weight = softmax(output[-1,:,:].squeeze().data.cpu().numpy())
    t = np.cumsum(weight)
    s = np.sum(weight)
    return int(np.searchsorted(t, np.random.rand(1)*s))

def sample_model(opt, model, hidden=None, prime_text=" ", length=1000, text=True): 

    # Make prime tensor
    text = [opt.vocab_ctoi.get(c, 0) for c in prime_text]
    prime_input = torch.LongTensor(text).view(-1, 1)
    prime_input = Variable(prime_input)
    
    # Make the list that stored every word
    gen_list = []
    gen_list.extend([char for char in prime_text])

    # Initialize state for generation
    if hidden is None:
        h = model.init_hidden()
    else:
        h = hidden
    # Add Cuda
    if opt.cuda:
        prime_input = prime_input.cuda()
    
    # Feed the model
    output, h = model(prime_input, h)

    # file to write
    f = open(os.path.join(opt.gen_dir, "result.txt"), 'w')
    
    # text to store
    gen_result = prime_text
    
    # Sample character by character
    for i in range(length):
        # Find what word is next and store to the list
        next_word = word_discriminator(output)
        # next_word = word_weight_discriminator(output)
        gen_list.append(opt.vocab_itoc[next_word])
        gen_result += opt.vocab_itoc[next_word]
        
        # Write the generated text to the file
        if len(gen_list) % 100 == 0:
            f.write(''.join(gen_list))
            f.flush()
            gen_list = []
        
        # Make input tensor using the last word generated
        char_input = Variable(torch.LongTensor([next_word]).view(-1, 1))
        if opt.cuda:
            char_input = char_input.cuda()
        
        output, h = model(char_input, h)
        
        # Need to repackage to disconnect the hidden nodes connection
        # if we don't use this the memory occupied by stack of hidden nodes will be too big
        h = h.detach()
        
    # After finishing generate write the remaining text to file
    f.write(''.join(gen_list))
    f.flush()
    f.close()
    return gen_result, h

if __name__ == "__main__":
    # get the information from argparse
    parser = argparse.ArgumentParser("Add information about making samples")
   
    parser.add_argument('--run', type=bool, default=False,
                        help="if you run main method")
    parser.add_argument('--epoch', type=int, default=None,
                        help="The epoch of saved model your gonna load")
    parser.add_argument('--prime', type=str, default=" ",
                        help="Enter the text that you wanna start with")
    parser.add_argument('--len', type=int, default=1000,
                        help="how long will you generate text?")
    parser.add_argument('--resume', default=False,
                        help="resume previous hidden state?")
    
    args = parser.parse_args()                         
    
    if args.run:
        assert args.epoch != None, "The epoch must be entered! --epoch [int]"
        
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
        hidden_size = 2048
        embedding_size = 2048
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
    
        # load the hidden data if resume is true
        if args.resume:
            with open(os.path.join(opt.gen_dir, "hidden.pkl"), 'rb') as f:
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
        save_path = os.path.join(opt.save_dir, "gru_{0}.pkl".format(args.epoch))
    
        # load saved model
        model.load_state_dict(torch.load(save_path))
    
        result, h = sample_model(opt=opt,
                              model=model,
                              hidden=h,
                              prime_text=args.prime,
                              length=args.len)
    
        # store the hidden state
        with open(os.path.join(opt.gen_dir, "hidden.pkl"), 'wb') as f:
            pickle.dump(h, f)
    
        print(result)

    
    

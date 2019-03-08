import os
import pickle
import torch
from tqdm import tqdm

from opt import opt
from train import train


def novel_list(path, path_list):
    filelist = os.listdir(path)
    for file in filelist:
        file = os.path.join(path, file)
        if os.path.isfile(file):
            path_list.append(file)
        elif os.path.isdir(file) and file[0] != '.':
            novel_list(file, path_list)


def highest_epoch(opt):
    epoch_list = os.listdir(opt.save_dir)
    epoch_list = [int(epoch[4:-4]) for epoch in epoch_list]
    highest_epoch = sorted(epoch_list)[-1]
    return highest_epoch
            
            
def multi_train(opt, path_list):
    for i, novel_path in enumerate(tqdm(path_list)):
        if i == 0:
            print("Started training on {}".format(novel_path.split('/')[-1]))
            opt.novel_path = novel_path
            opt.train = True
            opt.valid = True
            
            # Start training
            train(opt)
        else:
            print("Started training on {}".format(novel_path.split('/')[-1]))
            opt.novel_path = novel_path
            opt.resume = True
            opt.resume_epoch = highest_epoch(opt)
            opt.train = True
            opt.valid = True

            # Start training
            train(opt)

            
if __name__ == "__main__":
    # Directory
    # - Where is trainable data
    # - Where to save the model parameter
    # - Where to save the generated text
    data_dir = "data"
    save_dir = "save"
    gen_dir = "generate"

    # Choose the hyperparameter at here!
    ratio = 0.9
    num_layers = 2
    hidden_size = 2048
    embedding_size = 2048
    cuda = True if torch.cuda.is_available() else False
    batch_size = 32
    seq_len = 50
    num_epochs = 20
    save_every = 20
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
    
    path_list = []
    novel_list(opt.data_dir, path_list)
    multi_train(opt, path_list)

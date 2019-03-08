# -*- coding: utf-8 -*-

import os
import re
import pickle
import codecs


def novel_list(path, path_list):
    filelist = os.listdir(path)
    for file in filelist:
        file = os.path.join(path, file)
        if os.path.isfile(file):
            path_list.append(file)
        elif os.path.isdir(file) and file[0] != '.':
            novel_list(file, path_list)

            
def concat_novel(novel_list):
    concat_text = []
    for novel in novel_list:
        with codecs.open(novel, "r", encoding="utf-8") as f:
            text = f.read()

        if "</s>" in text:
            sentences = text.split("</s>")
        else:
            raise Exception('Please delimit the data with </s>')
            
        text = ' '.join(sentences)
        
        # Shorten the space width by regular expression
        text = re.sub('\s{2,}', repl=' ', string=text)
        text = [c for c in text]
        concat_text.extend(text)
    return concat_text


def gen_vocab(concat_text):
    vocab_list = ['<UNK>'] + sorted(list(set(concat_text)))
    return vocab_list
    
            
if __name__ == "__main__":
    
    # path list is where the novel paths are stored
    path_list = []
    path = 'data'
    
    # Check the what novel is in the data directory
    novel_list(path, path_list)
    
    # concatenate every novels in the list to one list
    concat_text = concat_novel(path_list)
    
    # Make dictionary
    vocab_list = gen_vocab(concat_text)
    
    vocab_size = len(vocab_list)
    vocab_itoc = {i: c for i, c in enumerate(vocab_list)}
    vocab_ctoi = {c: i for i, c in enumerate(vocab_list)}
    
    vocab_dic = {'vocab_size': vocab_size,
                 'vocab_itoc': vocab_itoc,
                 'vocab_ctoi': vocab_ctoi}

    with open(os.path.join('vocab', 'vocab.pkl'), 'wb') as f:
        pickle.dump(vocab_dic, f)



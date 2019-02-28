import torch.nn as nn
from torch.autograd import Variable


class gru(nn.Module):
    def __init__(self, opt):
        super(gru, self).__init__()

        # Parameters
        self.n_layers = opt.n_layers
        self.embed_size = opt.embedding_size
        self.hidden_size = opt.hidden_size
        self.vocab_size = opt.vocab_size
        self.batch_size = opt.batch_size

        # TODO - apply the facebook fasttext! And make it untrainable!
        self.emb = nn.Embedding(self.vocab_size, self.embed_size)

        # input size, hidden size
        # Changed batch_first=True -> False
        # Because of the dataParallel dimension problem
        self.gru = nn.GRU(self.embed_size,
                          self.hidden_size,
                          self.n_layers,
                          batch_first=False)
                          
        self.fc1 = nn.Linear(self.hidden_size, self.vocab_size)

    def forward(self, x, h):

        # Embedded layer
        x = self.emb(x)

        # GRU layer
        x, h = self.gru(x, h)

        # Linear layer
        x = self.fc1(x)

        return x, h

    def init_hidden(self):
        """Initialize hidden weights"""
        w = next(self.parameters()).data
        h = Variable(w.new(self.n_layers, self.batch_size, self.hidden_size).zero_())
        return h

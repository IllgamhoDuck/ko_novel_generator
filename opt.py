
class opt(object):
    def __init__(self, data_dir, save_dir, gen_dir,
                 ratio, num_layers, hidden_size, embedding_size,
                 cuda, batch_size, seq_len, num_epochs,
                 save_every, print_every, valid_every,
                 grad_clip, learning_rate):
        
        # The Parameter which has None will be filled while training
        self.resume = None
        self.resume_epoch = None
        
        # File Parameter
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.gen_dir = gen_dir
        self.novel_path = None
        
        # Model hyperparameter
        # vocab size must be fixed to facebook fast text
        # mode is used for data_loader - train / validate
        self.mode = None
        self.ratio = ratio
        self.n_layers = num_layers
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        
        # Training hyperparameter
        self.cuda = cuda
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_epochs = num_epochs
        self.save_every = save_every
        self.print_every = print_every
        self.valid_every = valid_every
        self.grad_clip = grad_clip
        self.lr = learning_rate
        
        # Vocabulary setting - This will be filled automatically
        self.vocab_size = None
        self.vocab_itoc = None
        self.vocab_ctoi = None
        
        # Check if we can use train / valid data
        self.train = True
        self.valid = True


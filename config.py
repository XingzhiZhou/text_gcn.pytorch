

class CONFIG(object):
    """docstring for CONFIG"""
    def __init__(self):
        super(CONFIG, self).__init__()
        
        self.dataset = 'R8'
        self.model = 'gin'  #dataset 'gcn', 'gcn_cheby', 'dense', 'gin'
        self.learning_rate = 0.02   # Initial learning rate.
        self.epochs  = 300  # Number of epochs to train.
        self.hidden1 = 200  # Number of units in hidden layer 1.
        self.dropout = 0.5  # Dropout rate (1 - keep probability).
        self.weight_decay = 0.   # Weight for L2 loss on embedding matrix.
        self.early_stopping = 80 # Tolerance for early stopping (# of epochs).
        self.max_degree = 3      # Maximum Chebyshev polynomial degree.

        self.num_layers = 2  # Number of layers in GIN
        self.num_mlp_layers = 1 #Number of layers in mlp
        self.embed_dim = 200 # embedding dimension
        self.hidden_dim_mlp = 200 # hidden layer dimension in mlp
        self.train_eps = False # decide whether include eps or not




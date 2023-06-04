import torch
from mylib.utils import *

class DataModule(HyperParameters):
    """The abstract class of data"""
    def __init__(self, root='../data', num_workers=4):
        self.save_hyperparameters()

    def train_dataloader(self):
        return self.get_dataloader(train=True)

    def val_dataloader(self):
        return self.get_dataloader(train=False)

    def get_tensorloader(self, tensors, train, indices=slice(0, None)):
        tensors = tuple(a[indices] for a in tensors)
        dataset = torch.utils.data.TensorDataset(*tensors)
        return torch.utils.data.DataLoader(dataset, self.batch_size,shuffle=train)

    
class SyntheticRegressionData(DataModule): 
    """Synthetic data generator for linear regression."""
    def __init__(self, w, b, noise=0.01, num_train=1000, num_val=1000, batch_size=32):
        super().__init__()
        self.save_hyperparameters() #saving already initialized values among parameters
        n = num_train + num_val
        self.X = torch.randn(n, len(w)) #design matrix X (of features)
        noise = torch.randn(n, 1) * noise
        self.y = torch.matmul(self.X, w.reshape((-1, 1))) + b + noise #vector of labels
        
    def get_dataloader(self, train):
        """Yields a minibatch of data at each next(iter(dataloader))"""
        # if train:
        #     indices = list(range(0, self.num_train))
        #     # The examples are read in random order
        #     random.shuffle(indices)
        # else:
        #     # Read in sequential order for debugging purposes
        #     indices = list(range(self.num_train, self.num_train+self.num_val))
        
        # for i in range(0, len(indices), self.batch_size):
        #     batch_indices = torch.tensor(indices[i: i+self.batch_size])
        #     yield self.X[batch_indices], self.y[batch_indices]
        
        i = slice(0, self.num_train) if train else slice(self.num_train, None)
        return self.get_tensorloader((self.X, self.y), train, i)
        
        


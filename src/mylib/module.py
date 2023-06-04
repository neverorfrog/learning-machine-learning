import torch
from torch import nn
from torch.nn import functional as F
from mylib.utils import *


class Module(nn.Module, HyperParameters):
    """The base class of models"""
    def __init__(self, plot_train_per_epoch = 2, plot_valid_per_epoch = 1):
        super().__init__()
        self.save_hyperparameters()
        self.board = ProgressBoard()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, X):
        assert hasattr(self, 'net'), 'Neural network is defined'
        return self.net(X)

    def plot(self, key, value, train):
        """Plot a point in animation."""
        assert hasattr(self, 'trainer'), 'Trainer is not inited'
        self.board.xlabel = 'epoch'
        if train:
            x = self.trainer.train_batch_idx / self.trainer.num_train_batches
            n = self.trainer.num_train_batches / self.plot_train_per_epoch
        else:
            x = self.trainer.epoch + 1
            n = self.trainer.num_val_batches / self.plot_valid_per_epoch
        self.board.draw(x, value.to(self.device).detach().numpy(),
                        ('train_' if train else 'val_') + key, every_n=int(n))
        #d2l.numpy(d2l.to(value, d2l.cpu()))
        #d2l.to(value, d2l.cpu()).detach().numpy()

    def training_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('loss', l, train=True)
        return l

    def validation_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('loss', l, train=False)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr)

    def apply_init(self, inputs, init=None):
        self.forward(*inputs)
        if init is not None:
            self.net.apply(init)
            
            
class LinearRegressionScratch(Module): 
    """The linear regression model implemented from scratch."""
    def __init__(self, num_weights, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.w = torch.normal(0, sigma, (num_weights, 1), requires_grad=True)
        self.b = torch.zeros(1, requires_grad=True)   

    #That's basically all our model amounts to when computing a label
    def forward(self, X):
        return torch.matmul(X, self.w) + self.b

    def loss(self, y_hat, y):
        l = ((y_hat - y) ** 2) / 2
        return l.mean()
    
    def configure_optimizers(self):
        return SGD([self.w, self.b], self.lr)

class LinearRegression(Module):
    """The linear regression model implemented with high-level APIs."""
    def __init__(self, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.LazyLinear(1)
        self.net.weight.data.normal_(0, 0.01)
        self.net.bias.data.fill_(0)

    def forward(self, X):
        return self.net(X)

    def loss(self, y_hat, y):
        fn = nn.MSELoss()
        return fn(y_hat, y)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), self.lr)

    def get_w_b(self):
        return (self.net.weight.data, self.net.bias.data)








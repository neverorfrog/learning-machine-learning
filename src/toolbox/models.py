import torch
from torch import nn
from toolbox.utils import HyperParameters
from toolbox.trainer import *
from toolbox.utils import *


class Module(nn.Module, HyperParameters):
    """The base class of models"""
    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, X):
        assert hasattr(self, 'net'), 'Neural network is defined'
        return self.net(X)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr)

    def apply_init(self, inputs, init=None):
        self.forward(*inputs)
        if init is not None:
            self.net.apply(init)
            
    def training_step(self, batch):
        #in the first argument of self.loss we call forward, passing it the unpacked batch of features
        l = self.loss(self(*batch[:-1]), batch[-1])
        return l
            

class LinearRegressionScratch(Module): 
    """The linear regression model implemented from scratch."""
    def __init__(self, input_dim, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.w = torch.zeros((input_dim, 1), requires_grad= True) 
        self.b = torch.zeros(1, requires_grad= True)

    #That's basically all our model amounts to when computing a label
    def forward(self, X):
        return torch.matmul(X, self.w) + self.b

    # The loss function is computed over all the samples in considered minibatch
    def loss(self, y_hat, y):
        l = torch.pow(y_hat - y, 2) / 2
        return (1 / self.trainer.batch_size) * l.sum()
    
    def training_step(self, batch):
        y_hat = self(*batch[:-1]) #prediction (calling forward)
        y = batch[-1] #labels
        self.optim_step(y_hat, y, *batch[:-1])
        return self.loss(y_hat,y)
    
    def optim_step(self, y_hat, y, X) -> None:
        error = (y_hat - y)
        n = len(self.w)
        m = self.trainer.batch_size
        
        dj_db = (1 / m) * error.sum()
        
        dj_dw = torch.zeros((n,1))
        for k in range(n):
            dj_dw[k] = (1 / m) * ((error * X[:,k]).sum()).item()
        
        self.w = self.w - self.lr * dj_dw
        self.b = self.b - self.lr * dj_db
    
    def configure_optimizers(self):
        return SGD([self.w, self.b], self.lr)
    
    def get_w_b(self):
        return (self.w, self.b)
        


class LinearRegression(Module):
    """The linear regression model implemented with high-level APIs."""
    def __init__(self, input_dim, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Linear(input_dim, 1, bias = True)
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
    
    
class LogisticRegressionScratch(Module): 
    """The logistic regression model implemented from scratch."""
    def __init__(self, input_dim, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.w = torch.ones((input_dim, 1), requires_grad= True) * 0.01 * (torch.rand(2).reshape(-1,1) - 0.5)
        self.b = torch.ones(1, requires_grad= True) * (-8)

    #That's basically all our model amounts to when computing a label
    def forward(self, X):
        return torch.sigmoid((torch.matmul(X,self.w) + self.b)).squeeze()

    # The loss function is computed over all the samples in the considered minibatch
    def loss(self, y_hat, y):
        y = y.type(torch.float32)
        l_one = torch.matmul(-y, torch.log(y_hat))
        l_zero = torch.matmul(-(1-y), torch.log(1-y_hat))
        return (l_one + l_zero)
    
    def training_step(self, batch):
        y_hat = self(*batch[:-1]) #prediction (calling forward)
        y = batch[-1] #labels
        self.optim_step(y_hat, y, *batch[:-1])
        return self.loss(y_hat,y)
    
    def optim_step(self, y_hat, y, X):
        error = (y_hat - y)
        n = len(self.w)
        m = self.trainer.batch_size
        
        dj_db = (1 / m) * error.sum()
        dj_dw = torch.tensor([(1/m) * (torch.matmul((y_hat - y),X[:,j])).sum() for j in range(n)]).reshape(-1,1)
        
        self.w = self.w - self.lr * dj_dw
        self.b = self.b - self.lr * dj_db
    
    def configure_optimizers(self):
        return SGD([self.w, self.b], self.lr)
    
    
class LogisticRegression(Module): 
    """The logistic regression model."""
    def __init__(self, input_dim, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = torch.nn.Linear(input_dim, 1)
        self.net.weight.data.normal_(0, 0.01)
        self.net.bias.data.fill_(0)

    #That's basically all our model amounts to when computing a label
    def forward(self, X):
        return torch.sigmoid(self.net(-X)).squeeze()

    # The loss function is computed over all the samples in the considered minibatch
    def loss(self, y_hat, y): 
        y = y.type(torch.float32) 
        return nn.BCELoss()(y_hat, y)
    
    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), self.lr)
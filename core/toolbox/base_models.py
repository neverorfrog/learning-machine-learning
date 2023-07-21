import torch
from torch import nn
from toolbox.utils import HyperParameters
from toolbox.trainer import *
from toolbox.utils import *
from torch.functional import F
from toolbox.math import *


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
        X = torch.tensor(*batch[:-1]) #features
        y_hat = self(X) #extraction of X and forward propagation
        y = batch[-1] #labels
        loss = self.loss(y_hat, y)
        self.trainer.plot('loss', loss, self.device, train = True)
        return loss

    def validation_step(self, batch):
        with torch.no_grad():
            loss = self.loss(self(*batch[:-1]), batch[-1])
        self.trainer.plot('loss', loss, self.device, train = False)
        

class Classifier(Module):
    """The base class of classification models"""
    def __init__(self):
        super().__init__()
        
    def loss(self, Y_hat, Y):
        return cross_entropy(Y_hat, Y)
    
    def predict(self, Y_hat):
        return Y_hat.argmax(axis=0)
         
    def validation_step(self, batch):
        with torch.no_grad():
            Y_hat = self(*batch[:-1])
            self.trainer.plot('loss', self.loss(Y_hat, batch[-1]), self.device, train=False)
            self.trainer.plot('acc', self.accuracy(Y_hat, batch[-1]), self.device, train=False)

    def accuracy(self, Y_hat, Y, averaged=True):
        """Compute the number of correct predictions"""
        Y_hat = Y_hat.reshape(-1, Y_hat.shape[-1]) # each column is a prediction for sample of belonging to each class
        predictions = self.predict(Y_hat).type(Y.dtype) # the most probable class is the one with highest probability
        compare = (predictions == Y.reshape(-1)).type(torch.float32) # we create a matrix of booleans 
        return compare.mean() if averaged else compare # fraction of ones wrt the whole matrix
    

class MLPScratch(Classifier):
    
    def __init__(self, input_dim, output_dim, hidden_dim, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W1 = torch.normal(0, sigma, size=(hidden_dim, input_dim),requires_grad=True)
        self.b1 = torch.zeros(size=(hidden_dim,1), requires_grad=True)
        self.W2 = torch.normal(0, sigma, size=(output_dim, hidden_dim),requires_grad=True)
        self.b2 = torch.zeros(size=(output_dim,1), requires_grad=True)
        
    def parameters(self):
        '''Parameters needed by the optimizer SGD'''
        return [self.W1, self.b1, self.W2, self.b2]
    
    def forward(self, X):
        X = (X.reshape((-1, self.input_dim))).T #one sample on each column -> X.shape = (d, m)
        a1 = relu(torch.matmul(self.W1, X) + self.b1)
        a2 = softmax(torch.matmul(self.W2, a1) + self.b2, dim = 0)
        return a2

class SoftmaxRegressionScratch(Classifier):
    def __init__(self, input_dim, output_dim, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W = torch.normal(0, sigma, size=(output_dim, input_dim),requires_grad=True)
        self.b = torch.zeros(size=(output_dim,1), requires_grad=True)

    def parameters(self):
        '''Parameters needed by the optimizer SGD'''
        return [self.W, self.b]
    
    def forward(self, X):
        X = (X.reshape((-1, self.input_dim))).T #one sample on each column -> X.shape = (d, m)
        Z = torch.matmul(self.W, X) + self.b
        predictions = softmax(Z, dim = 0) #softmax normalizes each row to one
        return predictions

        
    
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
    
    def training_step(self, batch) -> None:
        #Forward Propagation
        X = torch.tensor(*batch[:-1]) #features
        y_hat = self(X) #extraction of X and forward propagation
        y = batch[-1] #labels
        loss = self.loss(y_hat, y)
        self.trainer.plot('loss', loss, self.device, train = True)
        
        #Backward Propagation
        error = (y_hat - y)
        n = len(self.w) #number of features
        m = self.trainer.batch_size #number of examples
        
        dj_db = (1 / m) * error.sum()
        
        dj_dw = torch.zeros((n,1))
        for k in range(n):
            dj_dw[k] = (1 / m) * ((error * X[:,k]).sum()).item()
        
        self.w = self.w - self.lr * dj_dw
        self.b = self.b - self.lr * dj_db
    
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
    

class SoftmaxRegression(Classifier):
    """The softmax regression model"""
    def __init__(self, input_dim, output_dim, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, output_dim)
        )

    def forward(self, X):
        return self.net(X)
    
    # def loss(self, Y_hat, Y, averaged=True):
    #     Y_hat = Y_hat.reshape(-1, Y_hat.shape[-1])
    #     Y = Y.reshape(-1,)
    #     return F.cross_entropy(Y_hat, Y, reduction='mean' if averaged else 'none')
        
    
class LogisticRegressionScratch(Classifier): 
    """The logistic regression model implemented from scratch."""
    def __init__(self, input_dim, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        torch.manual_seed(1)
        self.w = 0.01 * (torch.rand(input_dim, requires_grad= True).reshape(-1,1) - 0.5)
        self.b = torch.ones(1, requires_grad= True) * (-8)
        
    #That's basically all our model amounts to when computing a label
    def forward(self, X):
        z = (torch.matmul(X,self.w) + self.b)
        return (1/(1 + torch.exp(-z))).squeeze() #since i know it's a vector, better having just one dim
    
    def predict(self, y_hat):
        return y_hat >= 0.5

    # The loss function is computed over all the samples in the considered minibatch
    def loss(self, y_hat, y):
        y = y.type(torch.float32)
        l_one = torch.matmul(-y, torch.log(y_hat))
        l_zero = torch.matmul(-(1-y), torch.log(1-y_hat))
        return torch.sum(l_one + l_zero) / self.trainer.batch_size
    
    def training_step(self, batch):
        #Forward Propagation
        X = torch.tensor(*batch[:-1]) #features
        y_hat = self(X) #extraction of X and forward propagation
        y = batch[-1] #labels
        loss = self.loss(y_hat, y)
        self.trainer.plot('loss', loss, self.device, train = True)
        
        #Backward Propagation
        error = (y_hat - y)
        n = len(self.w)
        m = self.trainer.batch_size
        
        dj_db = (1 / m) * torch.sum(error)
        dj_dw = torch.tensor([(1/m) * torch.sum(torch.matmul(error,X[:,j])) for j in range(n)]).reshape(-1,1)
        
        self.w = self.w - self.lr * dj_dw
        self.b = self.b - self.lr * dj_db

    def validation_step(self, batch):
        Y_hat = self(*batch[:-1])
        self.trainer.plot('loss', self.loss(Y_hat, batch[-1]), self.device, train=False)
        self.trainer.plot('acc', self.accuracy(Y_hat, batch[-1]), self.device, train=False)
    
    def configure_optimizers(self):
        return None
    
    
class LogisticRegression(Classifier): 
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
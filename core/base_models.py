import torch
from torch import nn
from core.utils import HyperParameters
from core.trainer import *
from core.utils import *


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
            
    def training_step(self,X,Y,plot = True): #forward propagation
        Y_hat = self(X) #shape = (m, d_out)
        loss = self.loss(Y_hat, Y) #loss computation
        if plot:
            self.trainer.plot('loss', loss, self.device, train = True)
        return loss

    def validation_step(self, X,Y, plot = True):
        with torch.no_grad():
            loss = self.loss(self(X), Y)
        if plot:
            self.trainer.plot('loss', loss, self.device, train = False)
        

class Classifier(Module):
    """The base class of classification models"""
    def __init__(self):
        super().__init__()
    
    def predict(self, X):
        return self(X).argmax(axis = 1).squeeze() #shape = (m)
         
    def validation_step(self,X,Y, plot = True):
        with torch.no_grad():
            accuracy = self.accuracy(X, Y)
            if plot:
                self.trainer.plot('acc', accuracy, self.device, train=False)
        return accuracy

    def accuracy(self, X, Y, averaged=True):
        """Compute the number of correct predictions"""
        predictions = self.predict(X).type(Y.dtype) # the most probable class is the one with highest probability
        compare = (predictions == Y).type(torch.float32) # we create a vector of booleans 
        return compare.mean() if averaged else compare # fraction of ones wrt the whole matrix

class SimpleNetwork(Classifier):
    def __init__(self, dimensions, loss = CrossEntropyLoss()):
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = dimensions[-1]
        self.net = nn.Sequential(nn.Flatten(), 
                                 nn.Linear(dimensions[0], dimensions[1]), nn.ReLU(), 
                                 nn.Linear(dimensions[1], dimensions[2]), nn.Softmax(dim = 1))
        
class NetworkSemiScratch(Classifier):
    
    def __init__(self, dimensions, lr, loss = CrossEntropyLoss(), sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = dimensions[-1]
        self.num_layers = len(dimensions) - 1   
        self.weights = [torch.normal(0, sigma, size=(dimensions[i-1], dimensions[i]),requires_grad=True) for i in range(1,len(dimensions))]
        self.biases = [torch.zeros(size=(1,dimensions[i]), requires_grad=True) for i in range(1,len(dimensions))]
        
    def parameters(self):
        '''Parameters needed by the optimizer SGD'''
        return [*self.weights, *self.biases]
    
    def forward(self, X):
        a = torch.flatten(X, start_dim = 1, end_dim = -1) #one sample on each row -> X.shape = (m, d)
        for i in range(self.num_layers-1):
            a = torch.sigmoid(torch.matmul(a, self.weights[i]) + self.biases[i])
        return softmax(torch.matmul(a, self.weights[-1]) + self.biases[-1], dim = 1)
        
class NetworkScratch(NetworkSemiScratch):
    
    def __init__(self, dimensions, lr, loss = CrossEntropyLoss(), sigma = 0.5, lmbda = 0.5):
        super().__init__(dimensions, lr, sigma)
        self.save_hyperparameters()
    
    def training_step(self,X,Y,n,plot = True):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single batch.
        ``n`` is the size of the entire dataset split"""
        loss = super().training_step(X,Y, plot)
        nabla_b, nabla_w = self.backprop(X, Y)
        self.weights = [(1 - self.lr*(self.lmbda/n))*w - (self.lr/len(X))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (self.lr/len(X))*nb
                       for b, nb in zip(self.biases, nabla_b)]
        return loss
        
    
    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of tensors, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [torch.zeros(b.shape) for b in self.biases]
        nabla_w = [torch.zeros(w.shape) for w in self.weights]
        
        # forward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = (torch.matmul(activation, w) + b) #shape = (m,n_i)
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
            
        # backward
        delta = activations[-1] 
        delta[range(len(x)),y.type(torch.long)] -= 1 #shape = (m,n_i)
        nabla_b[-1] = torch.sum(delta, dim = 0) #shape = (1,n_i)
        nabla_w[-1] = torch.matmul(activations[-2].T, delta) #shape = (n_{i-1}, n_i)
        for l in range(2, len(activations)):
            delta = torch.matmul(delta, self.weights[-l+1].T) * sigmoid_prime(zs[-l]) #shape = (1,n_i)
            nabla_b[-l] = torch.sum(delta, dim = 0) #shape = (1,n_i)
            nabla_w[-l] = torch.matmul(activations[-l-1].T, delta) #shape = (n_{i-1}, n_i) 
            
        return (nabla_b, nabla_w)

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
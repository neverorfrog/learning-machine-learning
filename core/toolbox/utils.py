import inspect
import torch

def softmax(z, dim):
    '''
    Returns matrix of same shape of z, but with the following changes:
    - all the elements are between 0 and 1
    - the sum of each slice along dim amounts to 1 
    '''
    expz = torch.exp(z)
    partition = expz.sum(dim, keepdim = True)
    return expz / partition

def cross_entropy(Y_hat, Y):
    """ Cross-entropy loss.
    Inputs:
    - Y (m,1): vector of indices for the correct class. -> m is batch size
    - Y_hat (m,d): predictions of the model. -> d is the number of classes
    Returns the average cross-entropy.
    """
    # This is called integer array indexing because the classes are modeled with one-hot encoding
    label_idx = Y.type(torch.int) #label for each example in the minibatch
    batch_idx = list(range(Y_hat.size(0))) #a list from 0 to m-1
    return -torch.log(Y_hat[batch_idx, label_idx]).mean()

def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)

def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+torch.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

class HyperParameters:
    def save_hyperparameters(self, ignore=[]):
        """Save function arguments into class attributes"""
        frame = inspect.currentframe().f_back
        _, _, _, local_vars = inspect.getargvalues(frame)
        self.hparams = {k:v for k, v in local_vars.items()
                        if k not in set(ignore+['self']) and not k.startswith('_')}
        for k, v in self.hparams.items():
            setattr(self, k, v)
            
class SGD(HyperParameters):
    """Minibatch stochastic gradient descent."""
    def __init__(self, params, lr):
        self.save_hyperparameters()

    def step(self):
        """That is fundamentally all learning amounts to in the end, namely modifying parameters
        of our hypothesis function (in this case linear with as many parameters as features)
        such that the loss function is minimized"""
        for param in self.params:
            param -= self.lr * param.grad  

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()



import inspect
import torch

class Parameters:
    def save_parameters(self, ignore=[]):
        """Save function arguments into class attributes"""
        frame = inspect.currentframe().f_back
        _, _, _, local_vars = inspect.getargvalues(frame)
        self.hparams = {k:v for k, v in local_vars.items()
                        if k not in set(ignore+['self']) and not k.startswith('_')}
        for k, v in self.hparams.items():
            setattr(self, k, v)


class QuadraticLoss:
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.

        """
        return 0.5 * torch.norm(a-y)**2

    def delta(z, a, y):
        """Return the error delta from the output layer."""
        return (a-y) * sigmoid_prime(z)


class CrossEntropyLoss:
    def __call__(self, a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.
            Inputs:
        - y (m,1): vector of indices for the correct class. -> m is batch size
        - a (m,d): activations (predictions) of the model. -> d is the number of classes

        """
        label_idx = y.type(torch.long) #label for each example in the minibatch
        batch_idx = list(range(a.size(0))) #a list from 0 to m-1
        return -torch.log(a[batch_idx, label_idx]).mean()

    def delta(z, a, y):
        """Return the error delta from the output layer.  Note that the
        parameter ``z`` is not used by the method.  It is included in
        the method's parameters in order to make the interface
        consistent with the delta method for other cost classes.

        """
        return (a-y)

def softmax(z, dim):
    '''
    Returns matrix of same shape of z, but with the following changes:
    - all the elements are between 0 and 1
    - the sum of each slice along dim amounts to 1 
    '''
    expz = torch.exp(z)
    partition = expz.sum(dim, keepdim = True)
    return expz / partition

def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)

def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+torch.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
            
class SGD(Parameters):
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



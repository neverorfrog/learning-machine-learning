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
    - Y (m,): vector of indices for the correct class. -> m is batch size
    - Y_hat (m, d): predictions of the model. -> d is the number of classes
    Returns the average cross-entropy.
    """
    # This is called integer array indexing because the classes are modeled with one-hot encoding
    row_idx = Y #label for each example in the minibatch
    column_idx = list(range(Y_hat.size(1))) #a list from 0 to m-1
    return -torch.log(Y_hat[row_idx, column_idx]).mean()

def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)
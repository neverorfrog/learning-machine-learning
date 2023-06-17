import torch


def softmax(X):
    '''
    Returns matrix of same shape of X, but with the following changes:
    - all the elements are between 0 and 1
    - the sum of each row amounts to 1 (on the rows we'll have the predictions for one sample)
    '''
    expx = torch.exp(X)
    partition = expx.sum(dim = 1, keepdim = True)
    return expx / partition

def cross_entropy(Y_hat, Y):
    """ Cross-entropy loss.
    Inputs:
    - Y (n,): vector of indices for the correct class. -> n is batch size
    - Y_hat (n, m): predictions of the model. -> m is the number of classes
    Returns the average cross-entropy.
    """
    return -torch.log(Y_hat[list(range(len(Y_hat))), Y]).mean()
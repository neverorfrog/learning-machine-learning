import numpy as np
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
import torch
from datamodule import Dataset
from trainer import Trainer
from utils import plot, softmax
from sklearn.metrics import classification_report, confusion_matrix

from sys import exit

class Classifier():
    
    def __init__(self, name):
        self.clf_map = {
            'K': [KNeighborsClassifier, 'KNN'],
            'S': [svm.LinearSVC, 'SVM']
            }

        self.clf = self.clf_map[name][0]() 
        
    def evaluate(self, dataset):
        preds = self.predict(dataset.X_test)
        print(classification_report(dataset.y_test, preds, digits=3))
        cm = confusion_matrix(dataset.y_test, preds, labels=None, sample_weight=None)
        print(cm)
        
    def accuracy(self, X, Y, averaged=True):
        """Compute the number of correct predictions"""
        predictions = torch.tensor(self.predict(X)).type(Y.dtype) # the most probable class is the one with highest probability
        compare = (predictions == Y).type(torch.float32) # we create a vector of booleans 
        return compare.mean() if averaged else compare # fraction of ones wrt the whole matrix
        
    def predict(self, X):
        return self.clf.predict(X)
        
    def fit(self, X, y):
        self.clf.fit(X, y)
        
class SoftmaxRegression(Classifier): 
    def __init__(self, num_features, num_classes, lr, sigma=0.01):
        torch.manual_seed(1)
        self.num_classes = num_classes #k = num_classes
        self.num_features = num_features # d = num_features
        self.W = torch.normal(0, sigma, size=(num_features, num_classes)) # shape = (d, k)
        self.b = torch.zeros(size=(1, num_classes)) # shape = (1,k)
        self.lr = lr
        
    #That's basically all our model amounts to when computing a label (together with predict function)
    def probs(self, X):
        # X: one sample on each row -> shape = (m, d)
        Z = torch.matmul(X, self.W) + self.b # one activation per sample on each column -> shape = (m, k)
        probabilities = softmax(Z, dim = 0) # softmax normalizes each column to one -> prob distribution
        return probabilities # shape = (m,k) -> probability distribution for one sample on each row

    
    def predict(self, X):
        return self.probs(X).argmax(axis = 1).squeeze() #shape = (m)

    # The loss function is computed over all the samples in the considered minibatch
    def loss(self, y_hat, y):
        y = y.type(torch.float32)
        """
        - y (m,1): vector of indices for the correct class. -> m is batch size
        - y_hat (m,k): predictions of the model

        """
        label_idx = y.type(torch.long) #label for each example in the minibatch (we are treating it as one-hot encoding)
        batch_idx = list(range(y_hat.size(0))) #a list from 0 to m-1
        loss = -torch.log(y_hat[batch_idx, label_idx]).mean() #scalar
        return loss
    
    def training_step(self, X, y, batch, plot = True):
        #Inference
        y_hat = self.probs(X) #extraction of X and forward propagation
        loss = self.loss(y_hat, batch[-1])
        if(plot):
            self.trainer.plot('loss', loss, train = True)
        else:
            print(loss)
        
        #Gradient Descent Step
        error = (y_hat - y) #shape = (m,k) -> each row represents how much the probability distribution diverges from the ideal one (one only in label pos)
        dj_db = torch.mean(error, dim = 0) # shape = (1,k)
        # torch.mean(error * X[:,j].unsqueeze(1), dim = 0) this gives me the derivative of cost wrt jth feature   
        dj_dw = torch.vstack([torch.mean(error * X[:,j].unsqueeze(1), dim = 0) for j in range(self.num_features)]) # shape = (d,k)
        self.W = self.W - self.lr * dj_dw
        self.b = self.b - self.lr * dj_db

    def test_step(self, batch):
        Y_hat = self.probs(*batch[:-1])
        self.trainer.plot('loss', self.loss(Y_hat, batch[-1]), self.device, train=False)
        self.trainer.plot('acc', self.accuracy(Y_hat, batch[-1]), self.device, train=False)
        
       
#ONLY FOR TWO CLASSES
class LogisticRegression(Classifier): 
    """The logistic regression model implemented from scratch."""
    def __init__(self, num_features, num_classes, lr, sigma=0.01):
        torch.manual_seed(1)
        self.w = 0.01 * (torch.rand(num_features).reshape(-1,1) - 0.5) #vector as tall as num_features
        self.b = torch.ones(1) * (-8) #scalar bias
        self.lr = lr
        
    #That's basically all our model amounts to when computing a label (together with predict function)
    def probs(self, X):
        X = torch.tensor(X, dtype=torch.float32)
        z = (torch.matmul(X,self.w) + self.b)
        return (1/(1 + torch.exp(-z))).squeeze() #since i know it's a vector, better having just one dim
    
    def predict(self, X):
        y_hat = self.probs(X)
        return y_hat >= 0.5

    # The loss function is computed over all the samples in the considered minibatch
    def loss(self, y_hat, y):
        y = y.type(torch.float32)
        l_one = torch.matmul(-y, torch.log(y_hat))
        l_zero = torch.matmul(-(1-y), torch.log(1-y_hat))
        return torch.sum(l_one + l_zero) / self.trainer.batch_size
    
    def training_step(self, X, y, batch, plot = True):
        #Inference
        X = torch.tensor(*batch[:-1]) #features
        y_hat = self.probs(X) #extraction of X and forward propagation
        y = batch[-1] #labels
        loss = self.loss(y_hat, y)
        if(plot):
            self.trainer.plot('loss', loss, train = True)
        else:
            print(loss)
        
        #Gradient Descent Step
        error = (y_hat - y)
        n = len(self.w)
        m = self.trainer.batch_size
        dj_db = (1 / m) * torch.sum(error)
        dj_dw = torch.tensor([(1/m) * torch.sum(torch.matmul(error,X[:,j])) for j in range(n)]).reshape(-1,1)
        self.w = self.w - self.lr * dj_dw
        self.b = self.b - self.lr * dj_db

    def test_step(self, batch):
        Y_hat = self.probs(*batch[:-1])
        self.trainer.plot('loss', self.loss(Y_hat, batch[-1]), self.device, train=False)
        self.trainer.plot('acc', self.accuracy(Y_hat, batch[-1]), self.device, train=False)
import numpy as np
from sklearn.model_selection import GridSearchCV

import torch
from datamodule import Dataset
from trainer import Trainer
from utils import plot, softmax
from sklearn.metrics import classification_report, confusion_matrix
from sys import exit

class Classifier():
    def __init__(self, clf):
        self.clf = clf
        
    def evaluate(self, dataset):
        preds = self.predict(dataset.X_test)
        print(classification_report(dataset.y_test, preds, digits=3))
        cm = confusion_matrix(dataset.y_test, preds, labels=None, sample_weight=None)
        print(cm)
        
    def score(self, X, Y, averaged=True):
        """Compute the number of correct predictions"""
        predictions = torch.tensor(self.predict(X)).type(Y.dtype) # the most probable class is the one with highest probability
        compare = (predictions == Y).type(torch.float32) # we create a vector of booleans 
        return compare.mean().item() if averaged else compare # fraction of ones wrt the whole matrix
        
    def predict(self, X):
        return self.clf.predict(X)
        
    def fit(self, X, y):
        self.clf.fit(X, y)

from sklearn.neighbors import KNeighborsClassifier    
class KNN(Classifier):
    def __init__(self, k = 5):
        super().__init__(KNeighborsClassifier(n_neighbors=k, n_jobs = -1))
        self.param_grid = {
            'n_neighbors': [8,20,50], 
            'weights': ['uniform', 'distance'],  # Different weight options
            'p': [1, 2]  # 1 for Manhattan, 2 for Euclidean
        }
        
from sklearn import svm
class SVM(Classifier):
    def __init__(self):
        super().__init__(svm.SVC(kernel="linear"))
        
        
from sklearn.naive_bayes import GaussianNB    
class GNB(Classifier):
    def __init__(self):
        super().__init__(GaussianNB())


from sklearn.linear_model import LogisticRegression
class LR(Classifier):
    def __init__(self):
        super().__init__(LogisticRegression(solver = "saga"))
            
class SoftmaxRegression(Classifier): 
    def __init__(self, num_features, num_classes, lr, sigma=0.5, lambda_ = 0.1):
        self.num_classes = num_classes #k = num_classes
        self.num_features = num_features # d = num_features
        self.W = torch.normal(0, sigma, size=(num_features, num_classes)) # shape = (d, k)
        self.b = torch.zeros(size=(1, num_classes)) # shape = (1,k)
        self.lr = lr
        self.lambda_ = lambda_
        
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
        regularizer = 0.5 * self.lambda_ * torch.sum(self.W)
        return loss + regularizer
    
    def training_step(self, X, y, batch, plot = True):
        #Inference
        y_hat = self.probs(X) #extraction of X and forward propagation
        loss = self.loss(y_hat, batch[-1])
        if(plot):
            self.trainer.plot('loss', loss, train = True)
            
        #error : shape = (m,k)
        #X: shape = (m,d)
        #W: shape = (d,k) 
        #Gradient Descent Step
        error = (y_hat - y) #shape = (m,k) -> each row represents how much the probability distribution diverges from the ideal one (one only in label pos)
        m = len(batch)
        b_grad = torch.mean(error, dim = 0) # shape = (1,k)
        # torch.mean(error * X[:,j].unsqueeze(1), dim = 0) this gives me the derivative of cost wrt jth feature   
        w_grad = torch.vstack([torch.mean(error * X[:,j].unsqueeze(1), dim = 0) for j in range(self.num_features)]) # shape = (d,k)
                
        self.W -= self.lr * w_grad + self.lambda_ * self.W
        self.b -= self.lr * b_grad

    def test_step(self, batch):
        Y_hat = self.probs(*batch[:-1])
        self.trainer.plot('loss', self.loss(Y_hat, batch[-1]), train=False)
        self.trainer.plot('acc', self.score(Y_hat, batch[-1]), train=False)
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Model(nn.Module):
    def __init__(self,num_classes,bias=True):
        super().__init__()
        #Convolutional Layers (take as input the image)
        self.conv1 = nn.Conv2d(3,32,kernel_size=8,stride=4,bias=bias,device=device)
        self.conv2 = nn.Conv2d(32,64,kernel_size=4,stride=2,bias=bias, device=device)
        self.conv3 = nn.Conv2d(64,64,kernel_size=2,stride=1,bias=bias, device=device)

        #Linear layers
        self.activation = nn.ReLU()
        self.linear1 = nn.Linear(64*9*9,64,bias=bias)
        self.linear2 = nn.Linear(64,32,bias=bias)
        self.linear3 = nn.Linear(32,num_classes,bias=bias)
        
        self.device = device

    def forward(self, x):
        '''
        Input:
            An image (already preprocessed)
        Output:
            Class of that image
        '''
        # Convolutions
        x = torch.tensor(x, dtype=torch.float32)
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        
        # Linear Layers
        # print("state shape={0}".format(x.shape)) #(32,64,8,8) or (1,64,8,8)
        x = torch.flatten(x,start_dim=1)
        # if (x.shape == (4096,64)): print("State={}".format(x.shape))
        x = self.activation(self.linear1(x))
        x = self.activation(self.linear2(x))
        y = self.linear3(x)
        # y = torch.softmax(y, dim = 1)
        # print("y"); print(y.shape) # (32,5)
        return y
    
    def loss(self, y_hat, y):
        fn = nn.CrossEntropyLoss()
        y = y.long()
        return fn(y_hat, y)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)
    
    def training_step(self,X,Y,plot = True): #forward propagation
        Y_hat = self(X) #shape = (m, d_out)
        loss = self.loss(Y_hat, Y) #loss computation
        if plot:
            self.trainer.plot('loss', loss, self.device, train = True)
        return loss
    
    def testing_step(self, X,Y, plot = True):
        with torch.no_grad():
            loss = self.loss(self(X), Y)
        if plot:
            self.trainer.plot('loss', loss, self.device, train = False)
            
    def predict(self, X):
        return self(X).argmax(axis = 1).squeeze() #shape = (m)
    
    def score(self, X, Y, averaged=True):
        """Compute the number of correct predictions"""
        predictions = torch.tensor(self.predict(X)).type(Y.dtype) # the most probable class is the one with highest probability
        compare = (predictions == Y).type(torch.float32) # we create a vector of booleans 
        return compare.mean().item() if averaged else compare # fraction of ones wrt the whole matrix
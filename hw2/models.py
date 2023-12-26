import os
import torch
import torch.nn as nn
from training_utils import Parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Classifier(nn.Module, Parameters):
    """The base class of models. Not instantiable because forward inference has to be defined by subclasses."""
    def __init__(self, name, num_classes, bias=True):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.save_parameters() #saves as class fields the parameters of the constructor

    def forward(self, X):
        pass

    def predict(self, X):
        return self(X).argmax(axis = 1).squeeze() #shape = (m)
    
    def save(self):
        path = os.path.join("models",self.name)
        if not os.path.exists(path): os.mkdir(path)
        torch.save(self.state_dict(), open(os.path.join(path,"model.pt"), "wb"))
        # print("MODELS SAVED!")

    def load(self, name):
        path = os.path.join("models",name)
        self.load_state_dict(torch.load(open(os.path.join(path,"model.pt"),"rb")))
        # print("MODELS LOADED!")
    

class CNN(Classifier):
    def __init__(self, name, num_classes, bias=True):
        super().__init__(name, num_classes, bias=True)
        
        #Channels
        self.channels0 = 3
        self.channels1 = 32
        self.channels2 = 64
        # self.channels3 = 64
        
        #Convolutional Layers (take as input the image)
        self.conv1 = nn.Conv2d(self.channels0, self.channels1, kernel_size=7, padding=1, stride=3, device=device)
        self.conv2 = nn.Conv2d(self.channels1, self.channels2, kernel_size=5, padding=1, stride=2, device=device)
        # self.conv3 = nn.Conv2d(self.channels2, self.channels3, kernel_size=3, padding=1, stride=1, device=device)
        
        #Linear layers
        self.activation = nn.ReLU()
        self.linear1 = nn.Linear(self.channels2*3*3,512,bias=bias)
        self.linear2 = nn.Linear(512,num_classes,bias=bias)
        
        #Max-pooling layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        #batch normalization layers
        self.batch_norm1 = nn.BatchNorm2d(self.channels1)
        self.batch_norm2 = nn.BatchNorm2d(self.channels2)
        # self.batch_norm3 = nn.BatchNorm2d(self.channels3)

        #dropout
        self.dropout = nn.Dropout(p=0.5)  # p is the probability of dropout

    def forward(self, x):
        '''
        Input:
            An image (already preprocessed)
        Output:
            Class of that image
        '''
        # Convolutions
        x = torch.tensor(x, dtype=torch.float32)
        x = self.batch_norm1(self.pool(self.activation(self.conv1(x))))
        x = self.batch_norm2(self.pool(self.activation(self.conv2(x))))
        # print("state shape={0}".format(x.shape))      
        # x = self.batch_norm3(self.pool(self.activation(self.conv3(x))))
        
        # Linear Layers
        x = torch.flatten(x,start_dim=1)
        x = self.activation(self.linear1(x))
        x = self.dropout(x)
        return self.linear2(x)
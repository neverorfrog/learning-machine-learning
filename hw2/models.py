import os
import torch
import torch.nn as nn
from training_utils import Parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from config import MODEL_PARAMS as params

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
        print("MODELS SAVED!")

    def load(self, name):
        path = os.path.join("models",name)
        self.load_state_dict(torch.load(open(os.path.join(path,"model.pt"),"rb")))
        self.eval()
        print("MODELS LOADED!")
    

class CNN(Classifier):
    def __init__(self, name, num_classes, bias=True):
        super().__init__(name, num_classes, bias=True)
        
        self.activation = nn.LeakyReLU()
        
        #Convolutional Layers (take as input the image)
        channels = params['channels']
        kernels = params['kernels']
        strides = params['strides']
        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=kernels[0], stride=strides[0], device=device)
        self.conv2 = nn.Conv2d(channels[0], channels[1], kernel_size=kernels[1], stride=strides[1], device=device)
        # self.conv3 = nn.Conv2d(channels[1], channels[2], kernel_size=kernels[2], stride=strides[2], device=device)
        
        #Linear layers
        self.linear1 = nn.Linear(channels[-1]*9*9,512,bias=bias)
        self.linear2 = nn.Linear(512,num_classes,bias=bias)
        
        #Max-pooling layers
        pool_kernels = params['pool_kernels']
        pool_strides = params['pool_strides']
        self.pool1 = nn.MaxPool2d(kernel_size=pool_kernels[0], stride=pool_strides[0])
        self.pool2 = nn.MaxPool2d(kernel_size=pool_kernels[1], stride=pool_strides[1])
        # self.pool3 = nn.MaxPool2d(kernel_size=pool_kernels[2], stride=pool_strides[2])
        
        #batch normalization layers
        self.batch_norm1 = nn.BatchNorm2d(channels[0])
        self.batch_norm2 = nn.BatchNorm2d(channels[1])
        # self.batch_norm3 = nn.BatchNorm2d(channels[2])

        #dropout
        self.dropout = nn.Dropout(params['dropout'])  # p is the probability of dropout

    def forward(self, x):
        '''
        Input:
            An image (already preprocessed)
        Output:
            Class of that image
        '''
        # Convolutions
        x = torch.tensor(x, dtype=torch.float32)
        x = self.activation(self.batch_norm1(self.pool1(self.conv1(x))))
        x = self.activation(self.batch_norm2(self.pool2(self.conv2(x))))
        # x = self.batch_norm3(self.pool3(self.activation(self.conv3(x))))
        # print("state shape={0}".format(x.shape))
        
        # Linear Layers
        x = torch.flatten(x,start_dim=1)
        x = self.activation(self.linear1(x))
        x = self.dropout(x)
        return self.linear2(x)
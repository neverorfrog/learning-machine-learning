import os
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
import torch.nn.functional as F
from training_utils import accuracy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Model(nn.Module):
    def __init__(self,name, num_classes, loss_function, score_function = accuracy, lr = 0.00001, bias=True):
        super().__init__()
        #Convolutional Layers (take as input the image)
        self.conv1 = nn.Conv2d(3,32,kernel_size=8,stride=4,bias=bias,device=device)
        self.conv2 = nn.Conv2d(32,64,kernel_size=4,stride=2,bias=bias, device=device)
        self.conv3 = nn.Conv2d(64,64,kernel_size=2,stride=1,bias=bias, device=device)

        #Linear layers
        self.activation = nn.ReLU()
        self.linear1 = nn.Linear(64,32,bias=bias)
        self.linear2 = nn.Linear(32,16,bias=bias)
        self.linear3 = nn.Linear(16,num_classes,bias=bias)
        
        #Normalization layers
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        
        #Max-pooling layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.device = device
        self.lr = lr
        self.name = name
        self.score_function = score_function
        self.num_classes = num_classes

    def forward(self, x):
        '''
        Input:
            An image (already preprocessed)
        Output:
            Class of that image
        '''
        # Convolutions
        x = torch.tensor(x, dtype=torch.float32)
        x = self.pool(self.activation(self.conv1(x)))
        x = self.pool(self.activation(self.conv2(x)))
        x = self.activation(self.conv3(x))
        
        # Linear Layers
        # print("state shape={0}".format(x.shape)) #(32,64,8,8) or (1,64,8,8)
        x = torch.flatten(x,start_dim=1)
        x = self.activation(self.linear1(x))
        x = self.activation(self.linear2(x))
        y = self.linear3(x)
        # print("y"); print(y.shape) # (32,5)
        return y
    
    def loss(self, y_hat, y):
        fn = nn.CrossEntropyLoss()
        y = y.long()
        return fn(y_hat, y)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr = self.lr, weight_decay = 0.001)
    
    def training_step(self,inputs,labels,plot = True): #forward propagation
        logits = self(inputs) #shape = (m, num_classes)
        loss = self.loss(logits, labels)
        if plot:
            self.trainer.plot('loss', loss, self.device, train = True)
        return loss
    
    def testing_step(self,inputs,labels,plot = True):
        with torch.no_grad():
            logits = self(inputs)
            loss = self.loss(logits, labels)
            predictions = torch.tensor(logits.argmax(axis = 1).squeeze()).type(labels.dtype) # the most probable class is the one with highest probability
            score = self.score_function(predictions,labels,self.num_classes)
        if plot:
            self.trainer.plot('loss', loss, self.device, train = False)
        return loss, score

    def predict(self, X):
        return self(X).argmax(axis = 1).squeeze() #shape = (m)
    
    def save(self):
        path = os.path.join("models",self.name)
        if not os.path.exists(path): os.mkdir(path)
        torch.save(self.state_dict(), open(os.path.join(path,"model.pt"), "wb"))
        # print("MODELS SAVED!")

    def load(self):
        path = os.path.join("models",self.name)
        self.load_state_dict(torch.load(open(os.path.join(path,"model.pt"),"rb")))
        # print("MODELS LOADED!")
        
    def evaluate(self, dataset):
        predictions_train = self.predict(dataset.X_train)
        predictions_test = self.predict(dataset.X_test)
        # evaluation against training set
        print("Train Score: ", self.score_function(predictions_train, dataset.y_train, self.num_classes))
        # evaluation against test set
        print("Test Score: ", self.score_function(predictions_test, dataset.y_test, self.num_classes))
        print(classification_report(dataset.y_test, predictions_test, digits=3))

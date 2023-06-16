import torch
from toolbox.utils import *
import pandas as pd
import numpy as np

class DataModule(HyperParameters):
    """The abstract class of data"""
    def __init__(self, root='../data', num_workers=4):
        self.save_hyperparameters()

    def train_dataloader(self):
        return self.get_dataloader(train=True)

    def val_dataloader(self):
        return self.get_dataloader(train=False)

    def get_tensorloader(self, tensors, train, indices=slice(0, None)):
        tensors = tuple(a[indices] for a in tensors)
        dataset = torch.utils.data.TensorDataset(*tensors)
        return torch.utils.data.DataLoader(dataset, self.batch_size, shuffle=train)

    
class SyntheticRegressionData(DataModule): 
    """Synthetic data generator for linear regression."""
    def __init__(self, w, b, noise=0.01, num_train=1000, num_val=1000, batch_size=32):
        super().__init__()
        self.save_hyperparameters(ignore=[w,b]) #saving already initialized values among parameters
        n = num_train + num_val #number of dataset samples
        self.X = torch.randn(n, len(w)) #design matrix X (of features)
        #b = torch.zeros(n, 1) #vector of bias values
        #self.X = torch.cat((b,self.X), dim = 1) #"augmented" design matrix with biases as first column
        noise = torch.randn(n, 1) * noise
        #self.w = torch.cat((torch.zeros((1,1)),self.w.reshape((-1, 1))),dim = 0)
        self.y = torch.matmul(self.X, self.w.reshape((-1, 1))) + b + noise #vector of labels
        
    def get_dataloader(self, train):
        """Yields a minibatch of data at each next(iter(dataloader))"""
        # if train:
        #     indices = list(range(0, self.num_train))
        #     # The examples are read in random order
        #     random.shuffle(indices)
        # else:
        #     # Read in sequential order for debugging purposes
        #     indices = list(range(self.num_train, self.num_train+self.num_val))
        
        # for i in range(0, len(indices), self.batch_size):
        #     batch_indices = torch.tensor(indices[i: i+self.batch_size])
        #     yield self.X[batch_indices], self.y[batch_indices]
        i = slice(0, self.num_train) if train else slice(self.num_train, None)
        return self.get_tensorloader((self.X, self.y), train, i)
    
class DataLoader(DataModule):
    
    def __init__(self, url, features, num_train, num_val, batch_size, class2Int = True):
        super().__init__()
        self.save_hyperparameters(ignore=[url]) #saving already initialized values among constructor parameters
        
        #Dataframe creation
        dataframe = pd.read_csv(url, names = features) #get datafrae from csv file
        self.dataframe = dataframe
        self.dataframe = dataframe
        self.initXy(dataframe)
        
    def __init__(self, path, num_train, num_val, batch_size, class2Int = True):
        super().__init__()
        self.save_hyperparameters(ignore=[path]) #saving already initialized values among constructor parameters
        
        #Dataframe creation
        dataframe = pd.read_csv(path) #get datafrae from csv file
        self.dataframe = dataframe
        self.initXy(dataframe)
    
    def __init__(self,txtfile):
        super().__init__()
        X, y = self.load_data(txtfile)
        self.X = torch.tensor(X).type(torch.float32)
        self.y = torch.tensor(y).type(torch.float32)
        self.num_train = len(y)
        self.num_val = 0
        self.batch_size = len(y)
    
    def initXy(self, dataframe):
        inputs, targets = dataframe.iloc[:, :-1], dataframe.iloc[:, -1]    
        #Features tensor
        self.X = torch.tensor(inputs.values).type(torch.float32)
        # self.b = torch.zeros(self.X.size(0), 1) #vector of bias values
        # self.X = torch.cat((b,X), dim = 1).type(torch.float32) #"augmented" design matrix with biases as first column
        
        #Label tensor
        y = np.array(targets.values)
        classNames = np.unique(y)
        classIndices = np.arange(0,len(classNames))
        classes = {className: classIndex for className, classIndex in zip(classNames, classIndices)}
        if self.class2Int:
            y = torch.tensor([classes[elem] for elem in y])
        else:
            y = torch.tensor(y)
        self.y = y

            
    def load_data(self, filename):      
        data = np.loadtxt(filename, delimiter=',')
        X = data[:,:-1]
        y = data[:,-1]
        return X, y
    
    def summarize(self):
        # gather details
        n_rows = self.X.shape[0]
        n_cols = self.X.shape[1]
        classes = np.unique(self.y)
        n_classes = len(classes)
        # summarize
        print(f'N Examples: {n_rows}')
        print(f'N Inputs: {n_cols}')
        print(f'N Classes: {n_classes}')
        print(f'Classes: {classes}')
        # class breakdown
        for c in classes:
            total = len(self.y[self.y == c])
            ratio = (total / float(len(self.y))) * 100
            print(f' - Class {str(c)}: {total} ({ratio})')
        if hasattr(self, 'dataframe'):
            self.dataframe.head()
        
    def get_dataloader(self, train):
        """Yields a minibatch of data"""
        i = slice(0, self.num_train) if train else slice(self.num_train, None)
        return self.get_tensorloader((self.X, self.y), train, i)
        


        
        


import colorsys
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
import torch

class Dataset():
    def __init__(self, path=None, num_train=None, num_test=None, batch_size=64, features=None,  X=None, y=None):
        #initializaing from CSV
        if path is not None:
            if features is not None:
                self.dataframe = pd.read_csv(path, names=features)
            else:
                self.dataframe = pd.read_csv(path)
            self.initXy(self.dataframe)
        #X and y already there
        elif X is not None and y is not None:
            self.X = X
            self.y = y
            
        self.classes = np.unique(self.y)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=1127)    
        self.batch_size = batch_size
        self.num_features = self.X.shape[1]
        self.num_train = len(self.y_train); self.num_test = len(self.y_test)
        self.train_data = tuple([self.X_train, self.y_train])
        self.test_data = tuple([self.X_test, self.y_test])
        self.num_classes = len(self.classes) 
              
    def train_dataloader(self):
        return self.get_dataloader(train=True)

    def test_dataloader(self):
        return self.get_dataloader(train=False)
    
    def get_dataloader(self, train):
        """Yields a minibatch of data at each next(iter(dataloader))"""
        data = self.train_data if train else self.test_data
        batch_size = self.batch_size if train else len(self.test_data)
        dataset = torch.utils.data.TensorDataset(*data)
        return torch.utils.data.DataLoader(dataset, self.batch_size, shuffle=train)
        
    def sparsity(self):
        index = np.mean(self.X <= 0.00001)
        return index
     
    def summarize(self):
        # gathering details
        n_rows = self.X.shape[0]
        n_cols = self.X.shape[1]
        n_classes = len(self.classes)
        # summarize
        print(f'N Examples: {n_rows}')
        print(f'N Inputs: {n_cols}')
        print(f'N Classes: {n_classes}')
        print(f'Classes: {self.classes}')
        # class breakdown
        for c in self.classes:
            total = len(self.y[self.y == c])
            ratio = (total / float(len(self.y))) * 100
            print(f' - Class {str(c)}: {total} ({ratio})')
        if hasattr(self, 'dataframe'):
            print(self.dataframe.head())
    
    def initXy(self, dataframe):       
        # Headers list:
        headers = dataframe.columns # 'x' for inputs, 'y' for labels
        
        #Inputs array
        inputs = dataframe[headers[0]]        
        # Convert an array-like string (e.g., '[0.02, 1.34\n, 2.12, 3.23\n]') 
        # into an array of floats (e.g., [0.02, 1.34, 2.12, 3.23]):
        inputs = [[float(feature) for feature in feature_vec.replace('[', '').replace(']', '').split()] for feature_vec in inputs]   
        #Features array
        self.X = torch.tensor(inputs)
                
        #Labels array
        if len(headers)>1:
            y_data = dataframe[headers[1]]
            y = torch.tensor(y_data)
        else:
            y = None      
        self.y = y
        
    def head(self):
        print(self.X[0])
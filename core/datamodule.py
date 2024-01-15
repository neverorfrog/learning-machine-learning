import random
import torch
from core.plotting_utils import show_images
from core.utils import *
import torchvision
import pandas as pd
from sklearn import datasets
import numpy as np
from torchvision import transforms
import numpy as np
import torch
import os
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from typing import Any, Self

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class Dataset(Parameters):
    """The abstract class of data"""
    def __init__(self,
    load: bool, 
    params: dict,
    train_data: Self = None,
    val_data: Self  = None,
    test_data: Self = None,
    path: str = None):
        self.save_parameters()
        self.load(path) if load else self.save()
        
    def save(self, path=None):
        if path is None: return
        torch.save(self.train_data, open(os.path.join(path,"train_data.dat"), "wb"))
        torch.save(self.val_data, open(os.path.join(path,"val_data.dat"), "wb"))
        torch.save(self.test_data, open(os.path.join(path,"test_data.dat"), "wb"))
        print("DATA SAVED!")
        
    def load(self, path=None):
        self.train_data = torch.load(open(os.path.join(path,"train_data.dat"),"rb"))
        self.val_data = torch.load(open(os.path.join(path,"val_data.dat"),"rb"))
        self.test_data = torch.load(open(os.path.join(path,"test_data.dat"),"rb"))
        print("DATA LOADED!\n")  
            
    def train_dataloader(self, batch_size):
        return self.get_dataloader(self.train_data, batch_size, use_weighting=self.params['use_weighted_sampler'])

    def val_dataloader(self, batch_size):
        return self.get_dataloader(self.val_data, batch_size, use_weighting=False)
    
    def get_dataloader(self, dataset, batch_size, use_weighting):
        """
        - Yields a minibatch of data at each next(iter(dataloader))
        - dataset needs to have __item__ 
        """
        #Stuff for weighted sampling
        weighted_sampler = None
        if use_weighting:
            weights = [self.params['class_weights'][label] for label in dataset.labels]
            weighted_sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        #Dataloader stuff
        g = torch.Generator()
        g.manual_seed(2000)
        return DataLoader(
            dataset,
            batch_size = batch_size,
            sampler = weighted_sampler,
            shuffle = not use_weighting,
            num_workers=12,
            worker_init_fn=seed_worker,
            generator=g,
        )
        
    def split_train(self, train_data, ratio):
        samples = train_data.samples
        labels = train_data.labels
        num_train_samples = len(samples)
        num_val_samples = int(ratio * num_train_samples)
        val_indices = np.random.permutation(num_train_samples)[:num_val_samples]
        
        #exclude samples that go into validation set
        new_train_data = samples[torch.tensor(list(set(range(num_train_samples)) - set(val_indices)))]
        new_train_labels = labels[torch.tensor(list(set(range(num_train_samples)) - set(val_indices)))]
        
        #data with sampled indices
        val_data = samples[val_indices]
        val_labels = labels[val_indices]
        
        return new_train_data, val_data, new_train_labels, val_labels
    
class ClassificationDataset(Dataset):
    def __init__(self,
    load: bool, 
    params: dict,
    train_data: Self = None,
    val_data: Self  = None,
    test_data: Self = None,
    path: str = None):
        super().__init__(load,params,train_data,val_data,test_data,path)
        self.classes = self.train_data.classes
            
    def summarize(self, split = None):
        # gathering data
        if split == 'train':
            data = self.train_data
        elif split == 'test':
            data = self.test_data
        elif split == 'val':
            data = self.val_data
        else:
            data = self
        
        # summarizing
        print(f'N Examples: {len(data.samples)}')
        print(f'N Classes: {len(data.classes)}')
        print(f'Classes: {data.classes}')
        
        # class breakdown
        for c in data.classes:
            total = len(data.labels[data.labels == c])
            ratio = (total / float(len(data.labels))) * 100
            print(f' - Class {str(c)}: {total} ({ratio})')
            
    
    
# class VisualDataset(DataModule):
#     def __init__(self, batch_size = 64):
#         super().__init__()
#         self.save_hyperparameters() #saving already initialized values among constructor parameters
#         X, y = datasets.make_moons(2000, noise=0.20)
#         separation = int(X.shape[0] * 0.8)
#         self.X_train = X[:separation]; self.X_test = X[separation:]             
#         self.y_train = y[:separation]; self.y_test = y[separation:]
#         self.num_train = len(self.y_train); self.num_test = len(self.y_test)
#         self.train_data = tuple([torch.tensor(self.X_train, dtype = torch.float32), torch.tensor(self.y_train, dtype = torch.float32)])
#         self.test_data = tuple([torch.tensor(self.X_test, dtype = torch.float32), torch.tensor(self.y_test, dtype = torch.float32)]) 
  
#     def get_dataloader(self, train):
#         """Yields a minibatch of data at each next(iter(dataloader))"""
#         data = self.train_data if train else self.test_data
#         dataset = torch.utils.data.TensorDataset(*data)
#         return torch.utils.data.DataLoader(dataset, self.batch_size, shuffle=train)

# class TorchDataset(DataModule):
#     def get_dataloader(self, train):
#         data = self.train_data if train else self.test_data
#         batch_size = self.batch_size if train else len(self.val_data)
#         return torch.utils.data.DataLoader(data, batch_size, shuffle=train, num_workers=self.num_workers)
        
#     def visualize(self, batch, nrows=1, ncols=8, labels=[]):
#         X, y = batch
#         if not labels:
#             labels = self.text_labels(y)
#         show_images(X.squeeze(1), nrows, ncols, titles=labels)
        
#     def val_split(self, train_data):
#         train_size = int(len(train_data) * 0.85)
#         val_size = int(len(train_data) - train_size)
#         train_data, val_data = torch.utils.data.random_split(train_data, [train_size, val_size])
#         return train_data, val_data        
        
#     def text_labels(self, indices):
#         """Return text labels"""
#         return [self.labels[int(i)] for i in indices]
        
# class MNIST(TorchDataset):
#     def __init__(self, batch_size=64, resize=(28, 28)):
#         super().__init__()
#         self.save_hyperparameters()
#         trans = transforms.Compose([transforms.Resize(resize),transforms.ToTensor()])
#         train_data = torchvision.datasets.MNIST(
#             root=self.root, train=True, transform=trans, download=True)
#         self.train_data, self.val_data = self.val_split(train_data)
#         self.test_data = torchvision.datasets.MNIST(
#             root=self.root, train=False, transform=trans, download=True)
#         self.labels = [0,1,2,3,4,5,6,7,8,9]

# class FashionMNIST(TorchDataset):
#     """The Fashion-MNIST dataset"""
#     def __init__(self, batch_size=64, resize=(28, 28)):
#         super().__init__()
#         self.save_hyperparameters()
#         trans = transforms.Compose([transforms.Resize(resize),transforms.ToTensor()])
#         self.train_data = torchvision.datasets.FashionMNIST(
#             root=self.root, train=True, transform=trans, download=True)
#         self.test_data = torchvision.datasets.FashionMNIST(
#             root=self.root, train=False, transform=trans, download=True)
#         self.labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
#                        'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
        
# class CSVDataset(DataModule):
#     def __init__(self, path = None, num_train = None, num_test = None, batch_size = None, features = None, class2Int = True):
#         super().__init__()
#         self.save_hyperparameters() #saving already initialized values among constructor parameters
#             #Dataframe creation
#         if features is not None:
#             dataframe = pd.read_csv(path, names = features) #get datafrae from csv file
#         else:
#             dataframe = pd.read_csv(path) 
#         self.dataframe = dataframe
#         self.initXy(dataframe)
    
#     def initXy(self, dataframe):
#         inputs, targets = dataframe.iloc[:, :-1], dataframe.iloc[:, -1]    
#         #Features tensor
#         self.X = torch.tensor(inputs.values).type(torch.float32)
#         self.b = torch.zeros(self.X.size(0), 1).type(torch.float32) #vector of bias values
        
#         #Label tensor
#         y = np.array(targets.values)
#         classNames = np.unique(y)
#         classIndices = np.arange(0,len(classNames))
#         classes = {className: classIndex for className, classIndex in zip(classNames, classIndices)}
#         if self.class2Int:
#             y = torch.tensor([classes[elem] for elem in y])
#         else:
#             y = torch.tensor(y)
#         self.y = y 
        

        
    


        


        
        


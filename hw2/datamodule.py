import numpy as np
from sklearn.model_selection import train_test_split
import torch
import os
import pandas as pd
from training_utils import *
from torch.utils.data import DataLoader, TensorDataset
from torchvision import io
from config import DATA_PARAMS as params
from torch.utils.data import WeightedRandomSampler

class Dataset():
    """
    path : folder containing the label folders
    """
    def __init__(self, load=False, X_train=None, y_train=None, X_test=None, y_test=None):
        #initializaing from folders of images
        if load is False:
            self.dataframe_train = self.create_dataframe("data", train = True)
            self.dataframe_test = self.create_dataframe("data", train = False)
            self.X_train, self.y_train = self.extract_tensors(self.dataframe_train)
            self.X_test, self.y_test = self.extract_tensors(self.dataframe_test)
            self.classes = np.unique(self.y_train)
            self.upsampling()
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, self.y_train, test_size=0.25)
            self.save()
        elif load is True:
            self.load()
            self.classes = np.unique(self.y_train)
            
        if X_train is not None and y_train is not None:
            self.X_train = X_train
            self.y_train = y_train
            self.X_test = X_test
            self.y_test = y_test
            
        self.num_features = self.X_train.shape[1]
        self.num_train = len(self.y_train)
        self.num_test = len(self.y_test)
        self.train_data = tuple([self.X_train, self.y_train])
        self.test_data = tuple([self.X_test, self.y_test])
        self.val_data = tuple([self.X_val, self.y_val])
        self.num_classes = len(self.classes)
    
    def create_dataframe(self,path,train):
        if train:
            path = path + "/train"
        else:
            path = path + "/test"
        
        data = []
        labels = []
        
        for label in os.listdir(path):
            label_folder = os.path.join(path, label)
            
            if os.path.isdir(label_folder):
                for image_file in os.listdir(label_folder):
                    image_path = os.path.join(label_folder, image_file)
                    data.append(io.read_image(image_path))
                    labels.append(label)
                
        df = pd.DataFrame({'Image': data, 'Label': labels})
        return df 
              
    def train_dataloader(self, batch_size):
        return self.get_dataloader(True, batch_size)

    def val_dataloader(self, batch_size):
        return self.get_dataloader(False, batch_size)
    
    def get_dataloader(self, train, batch_size):
        """Yields a minibatch of data at each next(iter(dataloader))"""
        data = self.train_data if train else self.val_data
        transform = params['train_transform'] if train else params['val_transform']
        labels = torch.tensor(data[-1], dtype=torch.int32)
        class_weights = params['train_class_weights'] if train else params['val_class_weights']
        weights = [class_weights[label] for label in labels]
        weighted_sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        if params['use_weighted_sampler'] == True:
            return DataLoader(
                dataset = TensorDataset(*data),
                batch_size = batch_size,
                sampler = weighted_sampler,
                collate_fn = lambda x: (
                    torch.stack([transform(item[0]) for item in x]),
                    torch.tensor([item[1] for item in x])
                ),
                num_workers=4
            )
        else:
            return DataLoader(
                dataset = TensorDataset(*data),
                batch_size = batch_size,
                shuffle=True,
                collate_fn = lambda x: (
                    torch.stack([transform(item[0]) for item in x]),
                    torch.tensor([item[1] for item in x])
                ),
                num_workers=4
            )
            
     
    def summarize(self, train = True):
        inputs = self.X_train if train is True else self.X_test
        labels = self.y_train if train is True else self.y_test
        
        # gathering details
        n_rows = inputs.shape[0] 
        n_cols = inputs.shape[1] 
        n_classes = len(self.classes)
        # summarize
        print(f'N Examples: {n_rows}')
        print(f'N Inputs: {n_cols}')
        print(f'N Classes: {n_classes}')
        print(f'Classes: {self.classes}')
        # print(f'Classe Weights: {self.class_weights}')
        # class breakdown
        for c in self.classes:
            total = len(labels[labels == c])
            ratio = (total / float(len(labels))) * 100
            print(f' - Class {str(c)}: {total} ({ratio})')
        if hasattr(self, 'dataframe'):
            print(self.dataframe.head())
    
    def extract_tensors(self, dataframe):       
        # Headers list:
        headers = dataframe.columns # 'x' for inputs, 'y' for labels
        #Inputs array
        images = dataframe[headers[0]] #this is a list
        
        #Features array
        X = torch.vstack([images[i].unsqueeze(0) for i in range(len(images))]) 
        
        #Labels array
        if len(headers)>1:
            labels = dataframe[headers[1]]
            y = torch.tensor(np.array(labels.values, dtype=np.float64))
        else:
            y = None
            
        return X,y     
        
    def head(self):
        print(self.X[0])
    
    def save(self):
        path = os.path.join("data","tensors")
        torch.save(self.X_train, open(os.path.join(path,"X_train.dat"), "wb"))
        torch.save(self.y_train, open(os.path.join(path,"y_train.dat"), "wb"))
        torch.save(self.X_val, open(os.path.join(path,"X_val.dat"), "wb"))
        torch.save(self.y_val, open(os.path.join(path,"y_val.dat"), "wb"))
        torch.save(self.X_test, open(os.path.join(path,"X_test.dat"), "wb"))
        torch.save(self.y_test, open(os.path.join(path,"y_test.dat"), "wb"))
        print("DATA SAVED!")

    def load(self):
        path = os.path.join("data","tensors")
        self.X_train = torch.load(open(os.path.join(path,"X_train.dat"),"rb"))
        self.y_train = torch.load(open(os.path.join(path,"y_train.dat"),"rb"))
        self.X_val = torch.load(open(os.path.join(path,"X_val.dat"),"rb"))
        self.y_val = torch.load(open(os.path.join(path,"y_val.dat"),"rb"))
        self.X_test = torch.load(open(os.path.join(path,"X_test.dat"),"rb"))
        self.y_test = torch.load(open(os.path.join(path,"y_test.dat"),"rb"))
        print("DATA LOADED!\n")  
        
    def upsampling(self):
        min_size = params['min_size_per_class']
        transform = params['train_transform']
        for label in self.classes:
            count = torch.sum(self.y_train == label).item()
            if count < min_size:
                difference = min_size - count
                inputs = self.extract_by_label(label)
                samples = [random.randint(0, count-1) for _ in range(difference)]
                upsampled_inputs = torch.vstack([inputs[samples[j]].unsqueeze(0) for j in range(difference)])
                upsampled_labels = torch.zeros(size=[difference]) + label
                self.X_train = torch.vstack([self.X_train, upsampled_inputs])
                self.y_train = torch.cat([self.y_train, upsampled_labels])
       
    def extract_by_label(self, label):
        class_mask = (self.y_train == label)
        return self.X_train[class_mask,:]
    
    def highest_label_count(self):
        highest_count = 0
        for label in self.classes:
            count = torch.sum(self.y_train == label).item()
            if count > highest_count:
                highest_count = count
        return highest_count
import numpy as np
import torch
import os
import pandas as pd
from image_utils import *
from training_utils import compute_class_weights
from torch.utils.data import DataLoader, WeightedRandomSampler, TensorDataset
from torchvision import transforms

class Dataset():
    """
    path : folder containing the label folders
    """
    def __init__(self, load=False, X_train=None, y_train=None, X_test=None, y_test=None):
        #initializaing from folders of images
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            random_color_jitter(),
            transforms.ToTensor()
        ])
        if load is False:
            self.dataframe_train = self.create_dataframe("data", train = True)
            self.dataframe_test = self.create_dataframe("data", train = False)
            self.X_train, self.y_train = self.extract_tensors(self.dataframe_train)
            self.X_test, self.y_test = self.extract_tensors(self.dataframe_test)
            self.save()
        elif load is True:
            self.load()
            
        if X_train is not None and y_train is not None:
            self.X_train = X_train
            self.y_train = y_train
            self.X_test = X_test
            self.y_test = y_test
            
        # assert(self.X_train)
            
        self.classes = np.unique(self.y_train)
        self.num_features = self.X_train.shape[1]
        self.num_train = len(self.y_train); self.num_test = len(self.y_test)
        self.train_data = tuple([self.X_train, self.y_train])
        self.test_data = tuple([self.X_test, self.y_test])
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
                    # All labels                          
                    data.append(process_image(image_path, self.transform))
                    labels.append(label)
                
        df = pd.DataFrame({'Image': data, 'Label': labels})
        return df 
              
    def train_dataloader(self, batch_size):
        return self.get_dataloader(True, batch_size)

    def test_dataloader(self, batch_size):
        return self.get_dataloader(False, batch_size)
    
    def get_dataloader(self, train, batch_size):
        """Yields a minibatch of data at each next(iter(dataloader))"""
        data = self.train_data if train else self.test_data
        labels = torch.tensor(data[-1], dtype=torch.int32)
        weights = [self.class_weights(labels)[label] for label in labels]
        weighted_sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        return DataLoader(
            dataset = TensorDataset(*data),
            batch_size = batch_size,
            sampler = weighted_sampler,
            # shuffle = train,
            collate_fn = lambda x: (
                torch.stack([self.transform(item[0]) for item in x]),
                torch.tensor([item[1] for item in x]))
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
        
    def class_weights(self, labels):
        class_weights = compute_class_weights(labels)
        return torch.tensor(class_weights, dtype=torch.float32)
    
    def save(self):
        path = os.path.join("data","tensors")
        torch.save(self.X_train, open(os.path.join(path,"X_train.dat"), "wb"))
        torch.save(self.y_train, open(os.path.join(path,"y_train.dat"), "wb"))
        torch.save(self.X_test, open(os.path.join(path,"X_test.dat"), "wb"))
        torch.save(self.y_test, open(os.path.join(path,"y_test.dat"), "wb"))
        print("DATA SAVED!")

    def load(self):
        path = os.path.join("data","tensors")
        self.X_train = torch.load(open(os.path.join(path,"X_train.dat"),"rb"))
        self.y_train = torch.load(open(os.path.join(path,"y_train.dat"),"rb"))
        self.X_test = torch.load(open(os.path.join(path,"X_test.dat"),"rb"))
        self.y_test = torch.load(open(os.path.join(path,"y_test.dat"),"rb"))
        print("DATA LOADED!\n")  
        
    def split_by_labels(self, zero=[0,3,4], one=[1,2]):
        """
        Params zero and one are labels (arrays) corresponding to the two classes
        """
        new_y_train = torch.tensor([0 if self.y_train[j] in zero else 1 for j in range(len(self.y_train))])
        new_y_test = torch.tensor([0 if self.y_test[j] in zero else 1 for j in range(len(self.y_test))])
        
        return Dataset(load = None, X_train = self.X_train, y_train = new_y_train, X_test = self.X_test, y_test = new_y_test)
        
        

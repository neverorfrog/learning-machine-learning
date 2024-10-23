import sys
sys.path.append('/home/flavio/code/main')

from src.datamodule import ClassificationDataset, Dataset
import numpy as np
import torch
import os
from torchvision import io
from config import DATA_PARAMS
from imblearn.combine import SMOTEENN

class ImageDataset(Dataset):
    def __init__(self, path=None, samples=None, labels=None, transform=None):
        self.save_parameters()
        if path is not None:
            samples = []
            labels = []
            
            for label in os.listdir(self.path):
                label_folder = os.path.join(self.path, label)
                
                if os.path.isdir(label_folder):
                    for image_file in os.listdir(label_folder):
                        image_path = os.path.join(label_folder, image_file)
                        item = io.read_image(image_path)
                        samples.append(item)
                        labels.append(int(label))
                        
            self.samples = torch.vstack([samples[i].unsqueeze(0) for i in range(len(samples))])
            self.labels = torch.tensor(labels, dtype=torch.int)
        self.classes = np.unique(self.labels)
    
    def __len__(self):
        return self.samples.shape[0]
    
    def __getitem__(self, index):
        img = self.samples[index]
        label = self.labels[index]
        
        if self.transform:
            img = self.transform(img)

        return img, label
    
        
class MyDataset(ClassificationDataset):
    def __init__(self, load: bool, params: dict, samples=None, labels=None):
        self.params = params
        #initializaing from folders of images
        if load is False:
            if samples is not None and labels is not None:
                train_dataset = ImageDataset(samples=samples,labels=labels)
                test_dataset = None
            else:
                #train split
                train_dataset = ImageDataset("data/train")                
                #test split
                test_dataset = ImageDataset("data/test")
                
                if self.params['resample']:
                    train_dataset = self.resample(train_dataset)
            
            # Train/Val Split
            train_samples, val_samples, train_labels, val_labels = self.split_train(train_dataset,ratio=self.params['val_split_size'])
            train_dataset = ImageDataset(samples=train_samples, labels=train_labels, transform=self.params['train_transform'])
            val_dataset = ImageDataset(samples=val_samples, labels=val_labels)        
            super().__init__(
                load=False,
                params=self.params,
                train_data=train_dataset, 
                val_data=val_dataset, 
                test_data=test_dataset)
        elif load is True:
            super().__init__(
                load=True,
                params=self.params)
            
    def resample(self,train_data):
        X = torch.flatten(train_data.samples,start_dim=1)
        y = train_data.labels
        combined = SMOTEENN(random_state=42)
        X_res, y_res = combined.fit_resample(X,y)
        X_res = torch.unflatten(torch.from_numpy(X_res), dim=1, sizes=[3,96,96])
        train_data = ImageDataset(samples=X_res, labels=y_res)
        return train_data
    
    def save(self):
        path = os.path.join("data","saved")
        super().save(path)

    def load(self):
        path = os.path.join("data","saved")
        super().load(path)

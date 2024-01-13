import numpy as np
import torch
import os
from training_utils import *
from torch.utils.data import DataLoader
from torchvision import io
from config import DATA_PARAMS as params
from torch.utils.data import WeightedRandomSampler
from PIL import Image

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class Dataset(Parameters):
    def __init__(self, load=False, train_data=None, val_data=None, test_data=None):
        self.load() if load is True else self.save_parameters()
        if load is False: self.save()
        self.classes = self.train_data.classes
        
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
              
    def summarize(self, split):
        # gathering data
        if split == 'train':
            data = self.train_data
        elif split == 'test':
            data = self.test_data
        elif split == 'val':
            data = self.val_data
        
        # summarizing
        print(f'N Examples: {len(data.samples)}')
        print(f'N Classes: {len(data.classes)}')
        print(f'Classes: {data.classes}')
        
        # class breakdown
        for c in data.classes:
            total = len(data.labels[data.labels == c])
            ratio = (total / float(len(data.labels))) * 100
            print(f' - Class {str(c)}: {total} ({ratio})')
        
    def train_dataloader(self, batch_size):
        return self.get_dataloader(self.train_data, batch_size, use_weighting=params['use_weighted_sampler'])

    def val_dataloader(self, batch_size):
        return self.get_dataloader(self.val_data, batch_size, use_weighting=False)
    
    def get_dataloader(self, dataset, batch_size, use_weighting):
        """
        - Yields a minibatch of data at each next(iter(dataloader))
        - dataset needs to have __item__ 
        """
        #Stuff for weighted sampling
        class_weights = params['train_class_weights']
        weights = [class_weights[label] for label in dataset.labels]
        weighted_sampler = WeightedRandomSampler(weights, len(weights), replacement=True) if use_weighting else None
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
    
    def upsampling(self, dataset):
        min_size = params['min_size_per_class']
        train_samples = dataset.samples
        train_labels = dataset.labels
        for label in dataset.classes:
            count = torch.sum(dataset.labels == label).item()
            if count < min_size:
                difference = min_size - count
                class_samples = train_samples[(train_labels == label),:]
                indices = [random.randint(0, count-1) for _ in range(difference)]
                
                new_samples = torch.vstack([class_samples[indices[j]].unsqueeze(0) for j in range(difference)])
                new_labels = torch.zeros(size=[difference], dtype=torch.int) + label
                
                train_samples = torch.vstack([train_samples, new_samples])
                train_labels = torch.cat([train_labels, new_labels])
                
        return train_samples, train_labels
        

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
    
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import EditedNearestNeighbours, RepeatedEditedNearestNeighbours
from imblearn.combine import SMOTEENN
        
class MyDataset(Dataset):
    def __init__(self, load=False, samples=None, labels=None):
        #initializaing from folders of images
        if load is False:
            if samples is not None and labels is not None:
                super().__init__(load=False, train_data=ImageDataset(samples=samples,labels=labels))
            else:
                #train split
                train_dataset = ImageDataset("data/train")
                train_samples, train_labels = self.upsampling(train_dataset)
                train_dataset = ImageDataset(samples=train_samples, labels=train_labels)
                
                #test split
                test_dataset = ImageDataset("data/test")
                
                #validation split
                train_samples, val_samples, train_labels, val_labels = self.split_train(train_dataset,ratio=params['val_split_size'])
                train_dataset = ImageDataset(samples=train_samples, labels=train_labels, transform=params['train_transform'])
                val_dataset = ImageDataset(samples=val_samples, labels=val_labels)
                
                if params['resample']:
                    train_dataset = self.resample(train_dataset)
                    
                super().__init__(load,train_data=train_dataset, val_data=val_dataset, test_data=test_dataset)
        elif load is True:
            super().__init__(load)
            
    def resample(self,train_data):
        X = torch.flatten(train_data.samples,start_dim=1)
        y = train_data.labels
        over = ADASYN(sampling_strategy='minority', random_state=42)
        under = RepeatedEditedNearestNeighbours(sampling_strategy='majority')
        X_res, y_res = over.fit_resample(X,y)
        # X_res, y_res = under.fit_resample(X,y)
        X_res = torch.unflatten(torch.from_numpy(X_res), dim=1, sizes=[3,96,96])
        train_data = ImageDataset(samples=X_res, labels=y_res)
        return train_data
        
    
    def save(self):
        path = os.path.join("data","saved")
        super().save(path)

    def load(self):
        path = os.path.join("data","saved")
        super().load(path)
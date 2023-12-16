import numpy as np
import torch
import cv2
import os
import pandas as pd
from sklearn.utils import shuffle

class Dataset():
    """
    path : folder containing the label folders
    """
    def __init__(self, path=None, num_train=None, num_test=None, batch_size=64, X=None, y=None):
        #initializaing from folders of images
        if path is not None:
            self.dataframe_train = self.create_dataframe(path, train = True)
            self.dataframe_test = self.create_dataframe(path, train = False)
            self.X_train, self.y_train = self.extract_tensors(self.dataframe_train)
            self.X_test, self.y_test = self.extract_tensors(self.dataframe_test)
            
        self.classes = np.unique(self.y_train)
        self.batch_size = batch_size
        self.num_features = self.X_train.shape[1]
        self.num_train = len(self.y_train); self.num_test = len(self.y_test)
        self.train_data = tuple([self.X_train, self.y_train])
        self.test_data = tuple([self.X_test, self.y_test])
        self.num_classes = len(self.classes)
        
    def process_image(self, image):
        return torch.tensor(image).permute(2, 0, 1)
    
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
                    
                    # Use the process_image function to read and process images
                    img = cv2.imread(image_path)                                
                    data.append(self.process_image(img))
                    labels.append(label)
                    
                    if label == '4' or label == '0':
                        new_img = self.brightness_contrast(img)
                        data.append(self.process_image(new_img))
                        labels.append(label)
                        
                    # if label == '4':
                    #     new_img = self.blur(img)
                    #     data.append(self.process_image(new_img))
                    #     labels.append(label)
                    
                
        df = pd.DataFrame({'Image': data, 'Label': labels})
        return df 
              
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
     
    def summarize(self):
        # gathering details
        n_rows = self.X_train.shape[0]
        n_cols = self.X_train.shape[1]
        n_classes = len(self.classes)
        # summarize
        print(f'N Examples: {n_rows}')
        print(f'N Inputs: {n_cols}')
        print(f'N Classes: {n_classes}')
        print(f'Classes: {self.classes}')
        # class breakdown
        for c in self.classes:
            total = len(self.y_train[self.y_train == c])
            ratio = (total / float(len(self.y_train))) * 100
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

    def brightness_contrast(self, image):
        alpha = 1.5  # Contrast control (1.0 means no change)
        beta = 50    # Brightness control (0 means no change)
        return cv2.addWeighted(image, alpha, np.zeros(image.shape, image.dtype), 0, beta)
    
    def blur(self, image):
        kernel_size = (5, 5)
        return cv2.GaussianBlur(image, kernel_size, 0) 

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class Dataset():
    def __init__(self, path = './data/dataset1.csv', num_train = None, num_test = None, batch_size = None, features = None, class2Int = True):
        #Dataframe creation
        if features is not None:
            dataframe = pd.read_csv(path, names = features) #get datafrae from csv file
        else:
            dataframe = pd.read_csv(path) 
        self.dataframe = dataframe
        self.initXy(dataframe)

     
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
            self.dataframe.head()
    
    def initXy(self, dataframe):       
        # Headers list:
        headers = dataframe.columns # 'x' for inputs, 'y' for labels
        
        #Inputs array
        inputs = dataframe[headers[0]]        
        # Convert an array-like string (e.g., '[0.02, 1.34\n, 2.12, 3.23\n]') 
        # into an array of floats (e.g., [0.02, 1.34, 2.12, 3.23]):
        inputs = [[float(feature) for feature in feature_vec.replace('[', '').replace(']', '').split()] for feature_vec in inputs]   
        #Features array
        self.X = np.array(inputs)
                
        #Labels array
        if len(headers)>1:
            y_data = dataframe[headers[1]]
            y = np.array(y_data)
        else:
            y = None      
        self.y = y
        self.classes = np.unique(self.y)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.333, random_state=1127)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split()
    
    
    
    
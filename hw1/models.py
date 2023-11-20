import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from datamodule import Dataset
from utils import plot

if __name__ == '__main__':
    dataset = Dataset()
    knn = KNeighborsClassifier(n_neighbors=51)
    knn.fit(dataset.X_train, dataset.y_train)
    preds = knn.predict(dataset.X_test)
    accuracy = np.mean(preds == dataset.y_test)
    print(accuracy)

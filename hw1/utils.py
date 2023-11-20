import colorsys
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
from sklearn.decomposition import PCA
from datamodule import Dataset
import matplotlib.colors as mcolors

def plot_train_data(dataset: Dataset):
    plot(dataset.X_train, dataset.y_train, dataset.classes)

def plot(dataset: Dataset, clf = None):
    # reducing dimension for graphical purposes
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(dataset.X_train)
    
    # axis limits
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    plt.axis([x_min,x_max,y_min,y_max])
    
    # generating points in this grid
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    points = np.c_[xx.ravel(), yy.ravel()]
    
    # predictions reshaped to match the meshgrid shape
    if clf:
        Z = clf.predict(pca.inverse_transform(points)).reshape(xx.shape)
    
    # plotting training points and decision boundaries
    colors, lighter_colors = get_colors(len(dataset.classes))
    my_map = mcolors.ListedColormap(colors)
    my_lighter_map = mcolors.ListedColormap(lighter_colors)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=dataset.y_train, cmap=my_map, marker='o', s=5, label='Data Points')
    if clf:
        plt.contourf(xx, yy, Z, alpha=0.15, cmap=my_map)
        plt.contour(xx,yy, Z, colors='k', linewidths=1)
    plt.show()


def get_colors(n, factor=0.5):
    np.random.seed(123)  # Set seed for reproducibility
    colors = plt.cm.rainbow(np.linspace(0, 1, n))
    lighter_colors = []
    if factor < 1:
        for color in colors:
            r, g, b = colorsys.rgb_to_hls(*matplotlib.colors.to_rgb(color))
            lighter_colors.append(colorsys.hls_to_rgb(r, g, min(1, factor * b)))
    return colors, lighter_colors
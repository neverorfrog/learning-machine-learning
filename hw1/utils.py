import collections
import colorsys
import inspect
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
from sklearn.decomposition import PCA
import torch
from datamodule import Dataset
import matplotlib.colors as mcolors
from IPython import display
from matplotlib_inline import backend_inline

class Parameters:
    def save_parameters(self, ignore=[]):
        """Save function arguments into class attributes"""
        frame = inspect.currentframe().f_back
        _, _, _, local_vars = inspect.getargvalues(frame)
        self.hparams = {k:v for k, v in local_vars.items()
                        if k not in set(ignore+['self']) and not k.startswith('_')}
        for k, v in self.hparams.items():
            setattr(self, k, v)

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

class ProgressBoard(Parameters):
    """The board that plots data points in animation."""
    
    def __init__(self, xlabel=None, ylabel=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 ls=['-', '--', '-.', ':'], colors=['C0', 'C1', 'C2', 'C3'],
                 fig=None, axes=None, figsize=(5, 4), display=True):
        self.save_parameters()
        self.fig = plt.figure(figsize=self.figsize)
        

    def draw(self, x, y, label, every_n=1):
                        
        #Creation of data structure for the points
        Point = collections.namedtuple('Point', ['x', 'y'])
        
        #ProgressBoard constructed for the first time
        if not hasattr(self, 'raw_points'):
            self.raw_points = collections.OrderedDict()
            self.data = collections.OrderedDict()
        
        #Label used for the first time
        if label not in self.raw_points:
            self.raw_points[label] = []
            self.data[label] = []
        points = self.raw_points[label]
        line = self.data[label]
        
        #Populating points dictionary with latest point
        points.append(Point(x, y))
        
        #Drawing a point only every n steps
        if len(points) != every_n:
            return
        
        #Adding a point to the line 
        mean = lambda x: sum(x) / len(x)
        line.append(Point(mean([p.x for p in points]),mean([p.y for p in points])))
        points.clear()
        
        #Display the line
        useSvgDisplay()
        plt_lines, labels = [], []
        for (k, v), ls, color in zip(self.data.items(), self.ls, self.colors):
            plt_lines.append(plt.plot([p.x for p in v], [p.y for p in v],linestyle=ls, color=color)[0])
            labels.append(k)
        axes = self.axes if self.axes else plt.gca()
        if self.xlim: axes.set_xlim(self.xlim)
        if self.ylim: axes.set_ylim(self.ylim)
        if not self.xlabel: self.xlabel = self.x
        axes.set_xlabel(self.xlabel)
        axes.set_ylabel(self.ylabel)
        axes.set_xscale(self.xscale)
        axes.set_yscale(self.yscale)
        axes.legend(plt_lines, labels)
        display.display(self.fig)
        display.clear_output(wait=True)
        
        
def useSvgDisplay():
    """Use the svg format to display a plot in Jupyter."""
    backend_inline.set_matplotlib_formats('svg')

def setFigsize(figsize=(3.5, 2.5)):
    """Set the figure size for matplotlib."""
    useSvgDisplay()
    plt.rcParams['figure.figsize'] = figsize

def setAxes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """Set the axes for matplotlib."""
    axes.set_xlabel(xlabel), axes.set_ylabel(ylabel)
    axes.set_xscale(xscale), axes.set_yscale(yscale)
    axes.set_xlim(xlim),     axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()


def softmax(z, dim):
    '''
    Returns matrix of same shape of z, but with the following changes:
    - all the elements are between 0 and 1
    - the sum of each slice along dim amounts to 1 
    '''
    expz = torch.exp(z)
    partition = expz.sum(dim, keepdim = True)
    return expz / partition


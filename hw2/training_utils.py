from matplotlib import pyplot as plt
import torch
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.discriminant_analysis import unique_labels

def softmax(z, dim):
    '''
    Returns matrix of same shape of z, but with the following changes:
    - all the elements are between 0 and 1
    - the sum of each slice along dim amounts to 1 
    '''
    expz = torch.exp(z)
    partition = expz.sum(dim, keepdim = True)
    return expz / partition

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    else:
        pass
        #print('Confusion matrix, without normalization')

    #print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    ax.set_ylim(len(classes)-0.5, -0.5)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax
    
    
def accuracy(predictions, labels, num_classes):
    """Compute the number of correct predictions"""
    compare = (predictions == labels).type(torch.float32) # we create a vector of booleans 
    return compare.mean().item() # fraction of ones wrt the whole matrix
    
    
def f1_score(predictions, labels, num_classes):
    """
    Calculate F1 score for multiclass classification.
    
    Args:
    - predictions (torch.Tensor)
    - labels (torch.Tensor)
    - num_classes (int)
    
    Returns:
    - f1_score (float)
    """
    f1_scores = []
    
    for class_label in range(num_classes):
        # Binary targets for the current class
        binary_targets = (predictions == labels).type(torch.float32)
        binary_predictions = (predictions == class_label).float()
        
        # True Positives, False Positives, False Negatives
        TP = torch.sum((binary_predictions == 1) & (binary_targets == 1)).item()
        FP = torch.sum((binary_predictions == 1) & (binary_targets == 0)).item()
        FN = torch.sum((binary_predictions == 0) & (binary_targets == 1)).item()
        
        # Precision, Recall
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        
        # F1 Score for the current class
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1_score)
    
    # Macro-Averaging: Average F1 scores across all classes
    macro_f1_score = sum(f1_scores) / num_classes if num_classes > 0 else 0.0
    
    return macro_f1_score


def sample_from_categorical(probabilities):
    """
    Samples from a categorical distribution using PyTorch.

    Parameters:
    - probabilities: A 1D tensor representing the probabilities of each category.

    Returns:
    - index: The index of the sampled category.
    """
    categorical_distribution = torch.distributions.Categorical(probabilities)
    sampled_index = categorical_distribution.sample()
    return sampled_index.item()
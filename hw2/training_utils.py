import inspect
import torch
import random
from torchvision import transforms
import torch

def random_color_jitter():
    # Randomly adjust brightness, contrast, saturation, and hue
    brightness_factor = random.uniform(0.8, 1.2)
    contrast_factor = random.uniform(0.8, 1.2)
    saturation_factor = random.uniform(0.8, 1.2)
    hue_factor = random.uniform(0.2, 0.5)

    color_jitter = transforms.ColorJitter(
        brightness=brightness_factor,
        contrast=contrast_factor,
        saturation=saturation_factor,
        hue=hue_factor
    )
    return color_jitter

class Standardization():
    def __init__(self):
        pass
    def __call__(self, sample):
        return sample / 255.0

class Parameters:
    def save_parameters(self, ignore=[]):
        """Save function arguments into class attributes"""
        frame = inspect.currentframe().f_back
        _, _, _, local_vars = inspect.getargvalues(frame)
        self.hparams = {k:v for k, v in local_vars.items()
                        if k not in set(ignore+['self']) and not k.startswith('_')}
        for k, v in self.hparams.items():
            setattr(self, k, v)

def compute_class_weights(targets):
    """
    Calculate class weights based on the provided targets.

    Parameters:
    - targets (torch.Tensor): 1D tensor containing class labels.
    - weighting_strategy (str): Strategy for calculating weights. Options: 'balanced'.

    Returns:
    - torch.Tensor: Computed class weights.
    """
    targets = torch.tensor(targets, dtype=torch.int32)
    class_counts = torch.bincount(targets)
    total_samples = class_counts.sum().float()

    # Compute weights to balance the classes
    weights = total_samples / (len(class_counts) * class_counts.float())

    # Normalize weights to sum to 1
    weights /= weights.sum()

    return weights

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
import torch
from training_utils import accuracy, f1_score, random_color_jitter
from torchvision import transforms
from torch import nn

DATA_PARAMS = {
    'train_class_weights': torch.tensor([0.24,0.1,0.1,0.12,0.48], dtype=torch.float32),
    'test_class_weights': torch.tensor([0.2,0.12,0.12,0.13,0.45], dtype=torch.float32),
    'train_transform': transforms.Compose([
        transforms.RandomRotation(20),
        transforms.RandomInvert(),
        transforms.RandomAutocontrast()
    ]),
    'test_transform': transforms.Compose([
    ])
}

TRAIN_PARAMS = {
    'max_epochs': 10,
    'learning_rate': 0.0001,
    'batch_size': 64,
    'patience': 5,
    'score_function': accuracy,
    'optim_function': torch.optim.Adam,
    'weight_decay': 0.1,
    'loss_function': nn.CrossEntropyLoss()
}
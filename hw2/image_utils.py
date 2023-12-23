import random
import cv2
import numpy as np
import torch
from torchvision import transforms


def process_image(image_path, transformation):
    img = cv2.imread(image_path)
    img = transformation(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  #Convert to RGB
    # img = img / 255.0  #Normalize pixel values to [0, 1]   
    return torch.tensor(img).permute(2, 0, 1)

def nothing(image):
    return image

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

def brightness_contrast(image):
    alpha = 1.5  # Contrast control (1.0 means no change)
    beta = 50    # Brightness control (0 means no change)
    return cv2.addWeighted(image, alpha, np.zeros(image.shape, image.dtype), 0, beta)

def blur(image):
    kernel_size = (5, 5)
    return cv2.GaussianBlur(image, kernel_size, 0) 
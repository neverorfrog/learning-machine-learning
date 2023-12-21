import cv2
import numpy as np
import torch


def process_image(image_path, transformation):
    img = cv2.imread(image_path)
    img = transformation(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  #Convert to RGB
    img = img / 255.0  #Normalize pixel values to [0, 1]   
    return torch.tensor(img).permute(2, 0, 1)

def nothing(image):
    return image

def brightness_contrast(image):
    alpha = 1.5  # Contrast control (1.0 means no change)
    beta = 50    # Brightness control (0 means no change)
    return cv2.addWeighted(image, alpha, np.zeros(image.shape, image.dtype), 0, beta)

def blur(image):
    kernel_size = (5, 5)
    return cv2.GaussianBlur(image, kernel_size, 0) 
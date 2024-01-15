import random
from torchvision import transforms

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


from typing import Tuple

import torch
import torchvision.transforms as transforms


class Preprocessor:

    def __init__(self, image_size: Tuple[int, int]):
        self.image_size = image_size

    def process(self, image: torch.Tensor):
        # Resize image
        transform = transforms.Resize(self.image_size, antialias=True)
        return transform(image)

import os
from typing import List, Tuple

import cv2
import torch
import numpy as np
from src.inference.source.AbstractLoader import Loader


class ImageLoader(Loader):

    def __init__(self, dir_path: str):

        self.dir_path: str = dir_path
        if not (os.path.exists(dir_path) and os.path.isdir(dir_path)):
            raise FileNotFoundError(f"Directory '{dir_path}' does not exist!")
        self.img_list: List[str] = os.listdir(self.dir_path)
        self.count: int = 0

    def get_fps(self) -> int | None:
        return None

    def has_next(self) -> bool:
        # Try to grab next frame
        return self.count < len(self.img_list)

    def next(self) -> Tuple[torch.Tensor, str | None] | None:
        # Read an image from the directory
        if not self.has_next():
            print("No more images in directory.")
            return None

        # Construct the path
        img_file: str = os.path.join(self.dir_path, self.img_list[self.count])

        # Read the image
        image: np.array = cv2.imread(img_file)

        # Convert the image to a Torch tensor
        image_tensor: torch.Tensor = torch.from_numpy(np.transpose(image, (2, 0, 1)))

        self.count += 1

        return image_tensor, img_file

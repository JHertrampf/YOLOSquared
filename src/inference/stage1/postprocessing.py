from typing import Tuple

import torch
import torchvision.transforms as T

from ultralytics.engine.results import Results


class Postprocessor:

    def __init__(self, patch_size: Tuple[int, int]):
        self.patch_size = patch_size
        self.transform = T.Resize(self.patch_size, antialias=True)

    def process(self, results: Results) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Bounding Boxes
        if len(results[0].boxes) == 0:
            return torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0)

        bboxes: torch.Tensor = results[0].boxes.xyxy

        # Confidences
        confs: torch.Tensor = results[0].boxes.conf

        # Classes
        cls: torch.Tensor = results[0].boxes.cls.int()

        # Patches
        original_img: torch.Tensor = results[0].orig_img

        # Loop over bboxes
        num_boxes: int = len(results[0].boxes)
        patches: torch.Tensor = torch.empty(num_boxes, 3, self.patch_size[0], self.patch_size[1])

        # Iterate over the bounding box coordinates and resize the patches
        for i, (x_1, y_1, x_2, y_2) in enumerate(results[0].boxes.xyxy):
            x_start, x_end = int(x_1), int(x_2)
            y_start, y_end = int(y_1), int(y_2)

            # Crop the image and resize the patch
            cropped_image: torch.Tensor = original_img[0, :, y_start:y_end, x_start:x_end]
            if cropped_image.size()[1] == 0 or cropped_image.size()[2] == 0:
                continue

            resized_image: torch.Tensor = self.transform(cropped_image)

            # Assign the resized patch to the patches tensor
            patches[i] = resized_image

        return bboxes, confs, cls, patches

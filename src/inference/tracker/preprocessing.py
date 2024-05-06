from typing import List

import numpy as np
import torch

class Preprocessor:

    def __init__(self):
        pass

    def process(self, bboxes_motorcycles: torch.Tensor, confs_motorcycles: torch.Tensor,
                cls_motorcycles: torch.Tensor, cls_helmets: List[torch.Tensor]) -> np.ndarray:
        if bboxes_motorcycles.numel() == 0:
            return np.empty((0, 6))

        # Convert the tensors to NumPy arrays
        bboxes: np.ndarray = bboxes_motorcycles.cpu().numpy()
        confs: np.ndarray = confs_motorcycles.cpu().numpy()
        cls_mc: np.ndarray = cls_motorcycles.cpu().numpy()
        cls_h: List[np.ndarray] = [cls.cpu().numpy() for cls in cls_helmets]

        classes: np.ndarray = np.empty_like(cls_mc, dtype=object)
        for i, class_h in enumerate(cls_h):
            # Concat with motorcycle classes
            classes[i] = f"{cls_mc[i]}.{','.join(class_h.astype(dtype=int).astype(dtype=str))}"

        # Extract the bounding box coordinates
        x1, y1, x2, y2 = bboxes.T

        # Stack the bounding box coordinates and confidences horizontally
        detections: np.ndarray = np.column_stack((x1, y1, x2, y2, confs, classes))

        return detections

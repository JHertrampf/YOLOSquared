from typing import Tuple, List

import numpy as np
import torch

from src.inference.tracker.postprocessing import Postprocessor
from src.inference.tracker.preprocessing import Preprocessor
from src.inference.tracker.sort import Sort


class Tracker:
    def __init__(self, max_age: int, min_hits: int, iou_threshold: float, min_class_update: int):
        self.preprocessor = Preprocessor()

        self.sort = Sort(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold, min_class_update=min_class_update)

        self.postprocessor = Postprocessor()

    def forward(self, bboxes_motorcycles: torch.Tensor, confs_motorcycles: torch.Tensor, cls_motorcycles: torch.Tensor,
                cls_helmets: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        # Preprocessing: Convert detections for SORT
        detections: np.ndarray = self.preprocessor.process(bboxes_motorcycles, confs_motorcycles, cls_motorcycles,
                                                           cls_helmets)

        # Apply tracker
        results: np.ndarray = self.sort.update(detections)

        # Postprocessing
        return self.postprocessor.process(results)

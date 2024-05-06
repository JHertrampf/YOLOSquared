from enum import Enum
from typing import List

import torch

from src.inference.head import AbstractHead
from src.inference.head.evaluation import Evaluator
from src.inference.head.visualization import Visualizer, VisualizationModes, SourceTypes


class HeadModes(Enum):
    EVALUATION = 0
    VISUALIZATION = 1


class Head:

    def __init__(self, mode: HeadModes, visualization_mode: VisualizationModes, source_type: SourceTypes,
                 output_path: str, show: bool):
        self.head: AbstractHead
        self.mode: HeadModes = mode
        if mode == HeadModes.EVALUATION:
            self.head = Evaluator(visualization_mode, output_path)
            self.output_path = output_path
        elif mode == HeadModes.VISUALIZATION:
            self.head = Visualizer(visualization_mode, source_type, output_path, show)

    def start(self) -> None:
        self.head.start()

    def forward(self, original_image: torch.Tensor, bboxes_motorcycles: torch.Tensor, confs_motorcycles: torch.Tensor,
                cls_motorcycles: torch.Tensor, bboxes_helmets: List[torch.Tensor], confs_helmet: List[torch.Tensor],
                cls_helmets: List[torch.Tensor], image_path: str = None, track_ids: torch.Tensor = None) -> None:
        if self.mode == HeadModes.VISUALIZATION:
            self.head.forward(original_image, bboxes_motorcycles, confs_motorcycles, cls_motorcycles, bboxes_helmets,
                              confs_helmet, cls_helmets, track_ids)
        elif self.mode == HeadModes.EVALUATION:
            self.head.forward(original_image, bboxes_motorcycles, confs_motorcycles, cls_motorcycles, bboxes_helmets,
                          confs_helmet, cls_helmets, image_path)
        else:
            raise NotImplementedError("Headmode not implemented.")

    def stop(self) -> None:
        self.head.stop()
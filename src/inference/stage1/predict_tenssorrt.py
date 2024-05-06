import os
from typing import Tuple

import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results


def create_engine(model_size: str, image_size: Tuple[int, int]) -> None:
    current_dir: str = os.path.dirname(os.path.abspath(__file__))
    model_pt: str = f"stage1_{model_size}.pt"
    model_engine: str = f"stage1_{model_size}.engine"

    if not os.path.exists(os.path.join(current_dir, model_engine)):
        # Create .engine from .pt file
        model = YOLO(os.path.join(current_dir, model_pt))
        model.export(format="engine", imgsz=image_size, device=torch.cuda.current_device(), simplify=True)


class PredictorTRT:
    def __init__(self, model_size: str, image_size: Tuple[int, int]):
        self.image_size: Tuple[int, int] = image_size
        self.model: YOLO

        # Build engine from .pt file if necessary
        create_engine(model_size, image_size)

        # Load engine for inference
        self.__load_engine(model_size)

    def __load_engine(self, model_size: str) -> None:
        current_dir: str = os.path.dirname(os.path.abspath(__file__))
        model_engine: str = f"stage1_{model_size}.engine"

        if not os.path.exists(os.path.join(current_dir, model_engine)):
            print("Missing .engine file for stage 1!")
        else:
            self.model = YOLO(os.path.join(current_dir, model_engine), task="detect")

    def forward(self, image: torch.Tensor) -> Results:
        # Add batch dimension and normalize
        # Need to clip as well because of numerical inaccuracies
        image_batched: torch.Tensor = torch.clip(torch.unsqueeze(image, dim=0) / 255, 0, 1)

        # Inference
        return self.model.predict(image_batched, agnostic_nms=True, imgsz=self.image_size)

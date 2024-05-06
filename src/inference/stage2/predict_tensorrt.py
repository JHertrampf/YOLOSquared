import os
from typing import Tuple

import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results


def create_engine(model_size: str, patch_size: Tuple[int, int]) -> None:
    current_dir: str = os.path.dirname(os.path.abspath(__file__))
    model_pt: str = f"stage2_{model_size}.pt"
    model_engine: str = f"stage2_{model_size}.engine"

    if not os.path.exists(os.path.join(current_dir, model_engine)):
        # Create .engine from .pt file
        model = YOLO(os.path.join(current_dir, model_pt))
        model.export(format="engine", device=torch.cuda.current_device(), dynamic=True, batch=100, imgsz=patch_size, simplify=True)


class PredictorTRT:
    def __init__(self, model_size: str, patch_size: Tuple[int, int]):
        self.model: YOLO
        self.patch_size: Tuple[int, int] = patch_size

        # Build engine from .pt file if necessary
        create_engine(model_size, patch_size)

        # Load engine for inference
        self.__load_engine(model_size)

    def __load_engine(self, model_size: str):
        current_dir: str = os.path.dirname(os.path.abspath(__file__))
        model_engine: str = f"stage2_{model_size}.engine"

        if not os.path.exists(os.path.join(current_dir, model_engine)):
            print("Missing .engine file for stage 2!")
        else:
            self.model = YOLO(os.path.join(current_dir, model_engine), task="detect")

    def forward(self, patches: torch.Tensor) -> Results:
        # Need to clip because of numerical inaccuracies
        patches: torch.Tensor = torch.clip(patches, 0, 1)

        # Inference
        return self.model.predict(patches, agnostic_nms=True, imgsz=self.patch_size)

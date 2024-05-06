import os

import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results


class PredictorULT:

    def __init__(self, model_size: str):
        """
        :param size: 'n' / 's' / 'm' / 'l' / 'x'
        """
        model_path: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"stage2_{model_size}.pt")
        self.model: YOLO = YOLO(model_path)

    def forward(self, patches: torch.Tensor) -> Results:
        # Need to clip because of numerical inaccuracies
        patches: torch.Tensor = torch.clip(patches, 0, 1)

        # Inference
        return self.model.predict(patches, agnostic_nms=True)

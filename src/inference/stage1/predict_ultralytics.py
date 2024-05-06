import os.path

from ultralytics import YOLO
import torch
from ultralytics.engine.results import Results


class PredictorULT:

    def __init__(self, model_size: str):
        """
        :param size: 'n' / 's' / 'm' / 'l' / 'x'
        """
        model_path: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"stage1_{model_size}.pt")
        self.model: YOLO = YOLO(model_path)

    def forward(self, image: torch.Tensor) -> Results:
        # Add batch dimension and normalize
        # Need to clip as well because of numerical inaccuracies
        image_batched: torch.Tensor = torch.clip(torch.unsqueeze(image, dim=0) / 255, 0, 1)

        # Inference
        return self.model.predict(image_batched, agnostic_nms=True)

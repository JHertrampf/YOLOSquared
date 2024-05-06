from typing import Tuple, List

import numpy as np
import torch


class Postprocessor:

    def __init__(self):
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def process(self, detections: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        bboxes: torch.Tensor = torch.from_numpy(detections[:, 0:4].astype(float)).to(self.device)
        track_ids: torch.Tensor = torch.from_numpy(detections[:, 4].astype(int)).to(self.device)
        cls_motorcycles: torch.Tensor = torch.tensor([int(x.split('.')[0]) for x in detections[:, 5]]).to(self.device)
        cls_helmets: List[torch.Tensor] = [torch.tensor(list(map(int, cls.split('.')[1].split(',')
        if cls.split('.')[1] != "" else []))).to(self.device) for cls in detections[:, 5]]
        return bboxes, track_ids, cls_motorcycles, cls_helmets

from abc import ABC, abstractmethod
from typing import List

import torch


class AbstractHead(ABC):

    def start(self) -> None:
        pass

    @abstractmethod
    def forward(self, original_image: torch.Tensor, bboxes_motorcycles: torch.Tensor, confs_motorcycles: torch.Tensor,
                cls_motorcycles: torch.Tensor, bboxes_helmets: List[torch.Tensor], confs_helmet: List[torch.Tensor],
                cls_helmets: List[torch.Tensor], track_ids: torch.Tensor = None) -> None:
        pass

    def stop(self) -> None:
        pass

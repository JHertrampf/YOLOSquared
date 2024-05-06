from abc import ABC, abstractmethod
from typing import Tuple

import torch


class Loader(ABC):
    def start(self) -> None:
        pass

    @abstractmethod
    def get_fps(self) -> int | None:
        pass

    @abstractmethod
    def has_next(self) -> bool:
        pass

    @abstractmethod
    def next(self) -> Tuple[torch.Tensor, str | None] | None:
        pass

    def stop(self) -> None:
        pass

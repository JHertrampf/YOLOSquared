from typing import Tuple

import cv2
import numpy as np
import torch

from src.inference.source.AbstractLoader import Loader


class CameraLoader(Loader):

    def __init__(self):
        self.cap = None

    def start(self) -> None:
        # Create a VideoCapture object for the default camera
        self.cap = cv2.VideoCapture(0)

    def get_fps(self) -> int | None:
        if self.cap is not None and self.cap.isOpened():
            return self.cap.get(cv2.CAP_PROP_FPS)

        return None

    def has_next(self) -> bool:
        # Try to grab next frame
        return self.cap.grab()

    def next(self) -> Tuple[torch.Tensor, str | None] | None:
        # Read a frame from the camera stream
        ret: bool
        frame: np.array
        ret, frame = self.cap.retrieve()
        if not ret:
            print("Error reading next frame!")
            return None

        # Mirror the frame on the horizontal axis
        mirrored_frame = cv2.flip(frame, 1)

        # Convert the mirrored frame to a Torch tensor
        image_tensor: torch.Tensor = torch.from_numpy(np.transpose(mirrored_frame, (2, 0, 1)))
        return image_tensor, None

    def stop(self) -> None:
        self.cap.release()

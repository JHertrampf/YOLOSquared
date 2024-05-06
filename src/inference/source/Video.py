import os
from typing import List, Tuple

import cv2
import numpy as np
import torch

from src.inference.source.AbstractLoader import Loader


def _get_video_files(video_dir: str) -> list[str]:
    video_files = []
    for file in sorted(os.listdir(video_dir)):
        if file.endswith(".mp4"):
            video_files.append(os.path.join(video_dir, file))
    return video_files


class VideoLoader(Loader):
    def __init__(self, video_dir: str):
        self.video_dir: str = video_dir
        self.video_files: List[str] = _get_video_files(video_dir)
        self.current_video_index: int = 0
        self.cap: cv2.VideoCapture = None
        self.grabbed_next: bool = False

    def _open_next_video(self) -> bool:
        if self.cap is not None:
            self.cap.release()

        if self.current_video_index < len(self.video_files):
            video_path: str = self.video_files[self.current_video_index]
            self.cap = cv2.VideoCapture(video_path)
            self.current_video_index += 1
            return True
        else:
            return False

    def start(self) -> None:
        self._open_next_video()

    def get_fps(self) -> int | None:
        if self.cap is not None and self.cap.isOpened():
            return self.cap.get(cv2.CAP_PROP_FPS)

        return None

    def has_next(self) -> bool:
        # Current video has frame or there are still other videos left
        self.grabbed_next = self.cap.grab()
        return self.grabbed_next or self.current_video_index < len(self.video_files)

    def next(self) -> Tuple[torch.Tensor, str | None] | None:
        if not self.grabbed_next:
            # Try to get next video and grab first frame
            if not (self._open_next_video() and self.has_next()):
                # No more videos
                return None

        ret, frame = self.cap.retrieve()
        if not ret:
            # Something went wrong
            print("Error reading next frame!")
            return None

        image: torch.Tensor = torch.from_numpy(np.transpose(frame, (2, 0, 1)))
        vid_path: str = self.video_files[self.current_video_index - 1]
        frame_no: int = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        file_path: str = vid_path.split(".mp4")[0] + "_frame_" + f"{frame_no:02}" + ".mp4"

        return image, file_path

    def stop(self) -> None:
        if self.cap is not None:
            self.cap.release()

from typing import Tuple

import cv2
import numpy as np
import torch
from goprocam import GoProCamera, constants
from goprocam.GoProCamera import GoPro
import netifaces as ni

from src.inference.source.AbstractLoader import Loader


def find_interface_port() -> int:
    # Find interface port
    interface_port = 0
    ifaces = ni.interfaces()
    for iface in ifaces:
        try:
            ip = ni.ifaddresses(iface)[ni.AF_INET][0]["addr"]
            if ip.startswith("172.") and "docker" not in iface:
                print(f"IP: {ip} from Interface {iface}")
                interface_port = iface
        except:
            print("No GoPro Camera found!")
    return interface_port


class GoProLoader(Loader):

    def __init__(self):
        # Find interface port
        interface_port: int = find_interface_port()

        # Get IP Address
        ip_address: str = GoProCamera.GoPro.getWebcamIP(interface_port)

        # Create GoPro
        self.gopro: GoPro = GoProCamera.GoPro(ip_address=ip_address, camera=constants.gpcontrol,
                                              webcam_device=interface_port)

        # Save UDP address of stream
        self.udp_address: str = "udp://@{}:8554".format(ip_address)

        self.cap: cv2.VideoCapture = None

    def start(self) -> None:
        self.gopro.startWebcam(resolution=constants.Webcam.Resolution.R1080p)
        self.gopro.webcamFOV(constants.Webcam.FOV.Linear)
        # Create a VideoCapture object for the UDP stream
        self.cap = cv2.VideoCapture(self.udp_address, cv2.CAP_FFMPEG)

    def get_fps(self) -> int | None:
        if self.cap is not None and self.cap.isOpened():
            return self.cap.get(cv2.CAP_PROP_FPS)

        return None

    def has_next(self) -> bool:
        # Try to grab next frame
        return self.cap.grab()

    def next(self) -> Tuple[torch.Tensor, str | None] | None:
        # Read a frame from the UDP stream
        ret: bool
        frame: np.array
        ret, frame = self.cap.retrieve()
        if not ret:
            print("Error reading next frame!")
            return None

        # Mirror the frame on the horizontal axis
        mirrored_frame: np.array = cv2.flip(frame, 1)

        # Convert the mirrored frame to a Torch tensor
        image_tensor: torch.Tensor = torch.from_numpy(np.transpose(mirrored_frame, (2, 0, 1)))
        return image_tensor, None

    def stop(self) -> None:
        self.cap.release()
        self.gopro.stopWebcam()

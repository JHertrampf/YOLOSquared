import cv2
import time
import argparse

import numpy as np

from src.inference.source.AbstractLoader import Loader
from src.inference.source.InternalCamera import CameraLoader
from src.inference.source.GoPro import GoProLoader
from src.inference.source.Video import VideoLoader
from src.inference.source.Images import ImageLoader


def main(loader_type: str, path: str = None):
    # Select the appropriate loader based on the input
    loader: Loader
    if loader_type == 'GoPro':
        loader = GoProLoader()
    elif loader_type == 'Internal':
        loader = CameraLoader()
    elif loader_type == 'Video':
        if path is None:
            raise ValueError("Video path is required when 'Video' loader is selected")
        loader = VideoLoader(path)
    elif loader_type == 'Images':
        if path is None:
            raise ValueError("Image path is required when 'Images' loader is selected")
        loader = ImageLoader(path)
    else:
        raise ValueError("Invalid loader type. Available options: 'GoPro', 'Internal', 'Video', 'Images'")

    # Start the loader
    loader.start()

    # Initialize variables for framerate calculation
    prev_time = time.time()
    frame_count = 0
    framerate = 0

    # Main loop to read and visualize frames
    while loader.has_next():
        # Get the next frame
        frame, _ = loader.next()

        # Check if there are no more frames
        if frame is None:
            break

        # Convert the torch tensor to a numpy array
        frame_np: np.ndarray = frame.numpy().transpose(1, 2, 0)

        # Calculate framerate
        current_time = time.time()
        frame_count += 1
        if current_time - prev_time >= 1.0:
            framerate = frame_count / (current_time - prev_time)
            frame_count = 0
            prev_time = current_time

        # Add framerate text to the frame
        framerate_text = "FPS: {:.2f}".format(framerate)
        text_color = (0, 255, 0)  # Bright green color
        text_position = (frame_np.shape[1] - 200, 50)  # Top right corner
        cv2.putText(frame_np, framerate_text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)

        # Display the frame using OpenCV
        cv2.imshow("Frame", frame_np)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Stop the loader
    loader.stop()

    # Close the OpenCV window
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Loader selection')
    parser.add_argument('--gopro', action='store_true', help='Use GoPro loader')
    parser.add_argument('--internal', action='store_true', help='Use default internal camera')
    parser.add_argument('--video', type=str, help='Path to the video file')
    parser.add_argument('--images', type=str, help='Path to the images folder')

    args = parser.parse_args()

    if args.gopro:
        main('GoPro')
    elif args.internal:
        main('Internal')
    elif args.video is not None:
        main('Video', args.video)
    elif args.images is not None:
        main('Images', args.images)
    else:
        parser.print_help()

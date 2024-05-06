from typing import List

import cv2
import numpy as np
import torch


def show_image(image: torch.Tensor, window_name: str, bboxes: List[torch.Tensor] = None,
               confs: List[torch.Tensor] = None, cls: List[torch.Tensor] = None) -> None:
    # Convert the Torch tensor to a NumPy array
    image_array = image.numpy()

    # Transpose the array to match the OpenCV format (height, width, channels)
    image_transposed = np.transpose(image_array, (1, 2, 0))

    # Iterate over the bounding boxes if present
    detections = 0
    if bboxes is not None and confs is not None and cls is not None:
        for (bboxes_mc, confs_mc, cls_mc) in zip(bboxes, confs, cls):
            for bbox, conf, cl in zip(bboxes_mc, confs_mc, cls_mc):
                detections += 1
                # Extract the coordinates of the current rectangle
                x1, y1, x2, y2 = bbox

                # Draw a rectangle on the image
                cv2.rectangle(image_transposed, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                # Add class and confidence
                text = f"{cl} {conf:.2f}"
                text_position = (int(x1), int(y1) - 10)  # Position the text above the rectangle
                cv2.putText(image_transposed, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Add amount of detections
        detections_text = f"Detections: {detections}"
        detections_position = (image_transposed.shape[1] - 150, 30)
        cv2.putText(image_transposed, detections_text, detections_position, cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 255), 2)

    # Display the image in a window
    cv2.imshow(window_name, image_transposed)

    # Wait for a key press to close the window
    cv2.waitKey(0)

    # Close all OpenCV windows
    cv2.destroyAllWindows()


def show_images_grid(image: torch.Tensor, window_name: str) -> None:
    # Convert the Torch tensor to a NumPy array
    image_array = image.numpy()

    # Reshape the image array to (batch_size, height, width, channels)
    image_array = np.transpose(image_array, (0, 2, 3, 1))

    # Determine the grid dimensions
    batch_size, height, width, channels = image_array.shape
    grid_size = int(np.ceil(np.sqrt(batch_size)))
    grid_height = grid_size * height
    grid_width = grid_size * width

    # Create an empty grid image
    grid_image = np.zeros((grid_height, grid_width, channels), dtype=np.uint8)

    # Fill the grid image with individual images and visualize bounding boxes
    for i in range(batch_size):
        row = i // grid_size
        col = i % grid_size
        start_y = row * height
        end_y = start_y + height
        start_x = col * width
        end_x = start_x + width
        grid_image[start_y:end_y, start_x:end_x] = image_array[i]

    # Display the grid image
    cv2.imshow(window_name, grid_image)

    # Wait for a key press to close the window
    cv2.waitKey(0)

    # Close all OpenCV windows
    cv2.destroyAllWindows()

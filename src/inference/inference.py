import csv
import os
import time
from time import sleep

import torch

from src.inference.head.head import HeadModes, Head
from src.inference.source import *
from src.inference.stage1 import *
from src.inference import inference_config
from src.inference.stage2 import Stage2
from src.inference.visualization_utils import show_image, show_images_grid
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'

def __compute_fps_metrics(fps_list: list):

    fps_min = min(fps_list)
    fps_max = max(fps_list)
    fps_avg = sum(fps_list) / len(fps_list)

    fps_sorted = sorted(fps_list)
    low_1_percent = sum(fps_sorted[:int(len(fps_sorted) * 0.01)]) / (len(fps_sorted) * 0.01)  # avg of lowest 1% of fps

    return fps_min, fps_max, fps_avg, low_1_percent


def main(data_source: inference_config.Sources, source_path: str, prediction_type: inference_config.Predictors,
         stage_1: inference_config.StageConfig, stage_2: inference_config.StageConfig,
         head: inference_config.HeadConfig, visualize=False):
    # Source
    source: AbstractLoader
    if data_source == inference_config.Sources.IMAGES:
        source = ImageLoader(source_path)
    elif data_source == inference_config.Sources.VIDEO:
        source = VideoLoader(source_path)
    elif data_source == inference_config.Sources.INTERNAL_CAMERA:
        source = CameraLoader()
    elif data_source == inference_config.Sources.GOPRO:
        source = GoProLoader()
    else:
        raise NotImplementedError(f"Source type '{data_source.name}' not implemented!")

    # Stage 1
    stage1: Stage1 = Stage1(stage_1.image_size, stage_2.image_size, stage_1.model_size, prediction_type)

    # Stage 2
    stage2: Stage2 = Stage2(stage_2.image_size, stage_2.model_size, prediction_type)

    # Head
    head: Head = Head(head.mode, head.visualization_mode, head.source_type, head.output_path, head.show)

    # set up fps documentation
    if inference_config.save_fps_data:
        fps_list = []
        num_detect_list = []

    # Setup
    source.start()
    head.start()

    # Inference
    while source.has_next():
        # Measure time for FPS synchronization
        frame_start_time: float = time.time()

        # Source
        image: torch.Tensor
        image_path: str
        image, image_path = source.next()

        if image is None:
            print("Error retrieving source!")
            break

        if image_path is None and head.mode == HeadModes.EVALUATION:
            print("Error retrieving source path!")
            break

        if visualize:
            show_image(image, "Source image")

        # Stage 1
        bboxes_motorcycles, confs_motorcycles, cls_motorcycles, patches_motorcycles = stage1.forward(image)

        if visualize:
            show_image(image, "Stage 1 Detections", [bboxes_motorcycles], [confs_motorcycles], [cls_motorcycles])
            show_images_grid(patches_motorcycles, "Extracted patches from Stage 1")

        # Stage 2
        bboxes_helmets, confs_helmet, cls_helmets = stage2.forward(patches_motorcycles, bboxes_motorcycles)

        if visualize:
            show_image(image, "Stage 2 Detections", bboxes_helmets, confs_helmet, cls_helmets)

        # Head
        head.forward(image, bboxes_motorcycles, confs_motorcycles, cls_motorcycles, bboxes_helmets, confs_helmet, cls_helmets, image_path)

        # Measure time for FPS synchronization
        frame_stop_time: float = time.time()
        frame_time: float = frame_stop_time - frame_start_time

        # Sleep for the rest of the target frame time to sync FPS with VIDEO
        target_fps: float = source.get_fps()
        if target_fps is not None:
            target_frame_time: float = 1 / target_fps
            sleep(max(target_frame_time - frame_time, 0))

        # document fps for evaluation
        if inference_config.save_fps_data:
            fps_list.append(1/frame_time)
            num_detect_list.append(len(bboxes_motorcycles) + len(bboxes_helmets))

    # Finish
    source.stop()

    # Compute fps information and write the data into a csv file
    if inference_config.save_fps_data:

        # remove the first entries as they are not meaningful for the metrics due to initialization time
        fps_list = fps_list[1:]
        num_detect_list = num_detect_list[1:]

        # compute the fps metrics
        min_fps, max_fps, avg_fps, min1percent_fps = __compute_fps_metrics(fps_list)

        # save the data to a csv file
        csv_file = inference_config.fps_csv_file
        with open(csv_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['FPS', '#Detect',  ' ', 'Min', 'Max', 'Avg', 'Min_1%'])  # Write header row
            for i, fps in enumerate(fps_list):  # Write data rows
                if i == 0:  # include the computed metrics in the first row
                    writer.writerow([fps, num_detect_list[i], " ", min_fps, max_fps, avg_fps, min1percent_fps])
                else:  # after the first row only include fps and the corresponding number of detections per prediction
                    writer.writerow([fps, num_detect_list[i], " ", " ", " ", " ", " "])
        print(f"CSV file with fps data saved to {csv_file}")


if __name__ == "__main__":
    main(inference_config.source, inference_config.source_path, inference_config.prediction_type,
         inference_config.stage_1, inference_config.stage_2, inference_config.head, visualize=False)

import os

import torch

from src.inference import inference_config
from src.inference.head import Head
from src.inference.source import *
from src.inference.stage1 import *
from src.inference.stage2 import Stage2
from src.inference.tracker.tracker import Tracker
from src.inference.visualization_utils import show_image, show_images_grid

os.environ['CUDA_MODULE_LOADING'] = 'LAZY'


def main(data_source: inference_config.Sources, source_path: str, prediction_type: inference_config.Predictors,
         stage_1: inference_config.StageConfig, stage_2: inference_config.StageConfig,
         head: inference_config.HeadConfig, sort: inference_config.SORTConfig, visualize=False):
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

    # SORT Tracker
    tracker: Tracker = Tracker(sort.max_age, sort.min_hits, sort.iou_threshold, sort.min_class_update)

    # Setup
    source.start()
    head.start()

    # Inference
    while source.has_next():

        # Source
        image: torch.Tensor
        image_path: str
        image, image_path = source.next()

        if image is None:
            print("Error retrieving source!")
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

        # Tracking
        bboxes_motorcycles_updated, track_ids, cls_motorcycles_updated, cls_helmets_updated = tracker.forward(
            bboxes_motorcycles, confs_motorcycles, cls_motorcycles, cls_helmets)

        # Head
        head.forward(image, bboxes_motorcycles_updated, confs_motorcycles, cls_motorcycles_updated, bboxes_helmets,
                     confs_helmet,
                     cls_helmets_updated, image_path, track_ids)

    # Finish
    source.stop()
    head.stop()


if __name__ == "__main__":
    main(inference_config.source, inference_config.source_path, inference_config.prediction_type,
         inference_config.stage_1, inference_config.stage_2, inference_config.head, inference_config.sort,
         visualize=False)

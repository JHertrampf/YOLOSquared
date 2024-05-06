from enum import Enum
from typing import Tuple
import config
from src.inference.head.head import HeadModes
from src.inference.head.visualization import VisualizationModes, SourceTypes


class Sources(Enum):
    IMAGES = 0
    VIDEO = 1
    INTERNAL_CAMERA = 2
    GOPRO = 3


class Predictors(Enum):
    ULTRALYTICS = 0
    TENSOR_RT = 1


class StageConfig:
    def __init__(self, image_size: Tuple[int, int], model_type: str):
        self.image_size = image_size  # [height, width]
        self.model_size = model_type  # n / s / m / l / x


class HeadConfig:
    def __init__(self, mode: HeadModes, visualization_mode: VisualizationModes, source_type: SourceTypes,
                 output_path: str, show: bool):
        self.mode = mode
        self.visualization_mode = visualization_mode
        self.source_type = source_type
        self.output_path = output_path
        self.show = show


class SORTConfig:
    def __init__(self, max_age: int, min_hits: int, iou_threshold: float, min_class_update: int):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.min_class_update = min_class_update


source = Sources.IMAGES
source_path = config.DATASET_ROOT_DIR
eval_output_path = "output"

prediction_type: Predictors = Predictors.TENSOR_RT
head_mode: HeadModes = HeadModes.VISUALIZATION

save_fps_data: bool = True
fps_csv_file: str = "fps"  # where to write fps data if save_fps_data = True

# configuration of the different parts of the inference pipeline:
stage_1: StageConfig = StageConfig((576, 1024), "l")
stage_2: StageConfig = StageConfig((224, 320), "s")
head: HeadConfig = HeadConfig(HeadModes.VISUALIZATION, VisualizationModes.MINIMAL, SourceTypes.CONTINUOUS, "output", False)
sort: SORTConfig = SORTConfig(1, 1, 0.3, 3)

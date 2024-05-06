from typing import Tuple

from ultralytics.engine.results import Results

from src.inference.stage1.preprocessing import Preprocessor
from src.inference.stage1.predict_ultralytics import PredictorULT
from src.inference.stage1.predict_tenssorrt import PredictorTRT
from src.inference.stage1.postprocessing import Postprocessor
from ..inference_config import Predictors

import torch


class Stage1:

    def __init__(self, image_size: Tuple[int, int], patch_size: Tuple[int, int], model_size: str,
                 predictor: Predictors):

        # Initialize preprocessor
        self.preprocessor = Preprocessor(image_size)

        # Initialize predictor
        if predictor == Predictors.ULTRALYTICS:
            self.predictor = PredictorULT(model_size)
        elif predictor == Predictors.TENSOR_RT:
            self.predictor = PredictorTRT(model_size, image_size)
        else:
            raise NotImplementedError(f"Prediction type '{predictor.name}' not implemented!")

        # Initialize postprocessor
        self.postprocessor = Postprocessor(patch_size)

    def forward(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        # Preprocessing
        processed_img: torch.Tensor = self.preprocessor.process(img)

        # Prediction:
        results: Results = self.predictor.forward(processed_img)

        # Postprocessing
        return self.postprocessor.process(results)

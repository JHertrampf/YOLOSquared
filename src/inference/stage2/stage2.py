from typing import Tuple, List

import torch
from ultralytics.engine.results import Results

from ..inference_config import Predictors
from .preprocessing import Preprocessor
from .predict_ultralytics import PredictorULT
from .predict_tensorrt import PredictorTRT
from .postprocessing import Postprocessor


class Stage2:

    def __init__(self, patch_size: Tuple[int, int], model_size: str,
                 predictor: Predictors):

        self.preprocessor = Preprocessor()

        # Initialize predictor
        if predictor == Predictors.ULTRALYTICS:
            self.predictor = PredictorULT(model_size)
        elif predictor == Predictors.TENSOR_RT:
            self.predictor = PredictorTRT(model_size, patch_size)
            pass
        else:
            raise NotImplementedError(f"Prediction type '{predictor.name}' not implemented!")

        self.postprocessor = Postprocessor(patch_size)

    def forward(self, patches: torch.Tensor, bboxes_motorcycles: torch.Tensor) \
            -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:

        # Preprocessing
        processed_patches = self.preprocessor.process(patches)

        # Prediction
        if len(processed_patches) == 0:  # no motorcycle patches found
            return [], [], []
        results: Results = self.predictor.forward(processed_patches)

        # Postprocessing
        return self.postprocessor.process(results, bboxes_motorcycles)

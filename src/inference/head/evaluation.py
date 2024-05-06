from typing import List

import torch
import os

from src.inference.head.AbstractHead import AbstractHead
from src.inference.head.visualization import VisualizationModes


class Evaluator(AbstractHead):

    def __init__(self, visualization_mode: VisualizationModes, output_path: str):
        self.visualization_mode = visualization_mode
        self.mode_str = visualization_mode.name
        self.output_path = output_path


    def forward(self, original_image: torch.Tensor, bboxes_motorcycles: torch.Tensor, confs_motorcycles: torch.Tensor,
                cls_motorcycles: torch.Tensor, bboxes_helmets: List[torch.Tensor], confs_helmet: List[torch.Tensor],
                cls_helmets: List[torch.Tensor], image_path: str):

        file_name = os.path.splitext(os.path.basename(image_path))[0] + ".txt"  # file to write prediction into

        # Determine amount of helmets
        helmet_counts: List[int] = [torch.sum(tensor == 1).item() for tensor in cls_helmets]

        # Visualize results in image depending on mode
        if self.visualization_mode == VisualizationModes.FULL:

            self.write_detections(bboxes_motorcycles, cls_motorcycles, confs_motorcycles, file_name, bboxes_helmets, cls_helmets, confs_helmet)

        elif self.visualization_mode == VisualizationModes.SIMPLE:

            mc_labels = []
            for i, patch in enumerate(bboxes_motorcycles):
                class_num = int(cls_motorcycles[i].item())
                cur_label = f"{class_num}p_{helmet_counts[i]}h"
                mc_labels.append(cur_label)

            self.write_detections(bboxes_motorcycles, mc_labels, confs_motorcycles, file_name)

        elif self.visualization_mode == VisualizationModes.MINIMAL:

            violation: torch.Tensor = torch.full((len(cls_helmets),), False, dtype=torch.bool)
            violation_labels = []

            for i, patch_classes in enumerate(cls_helmets):
                violation[i] = cls_motorcycles[i] > helmet_counts[i] or cls_motorcycles[i] > 3

            for i, patch in enumerate(bboxes_motorcycles):
                if violation[i]:
                    cur_violation = "VIOLATION"
                    violation_labels.append(cur_violation)
                else:
                    cur_violation = "okay"
                    violation_labels.append(cur_violation)

            self.write_detections(bboxes_motorcycles, violation_labels, confs_motorcycles, file_name)

        else:
            raise NotImplementedError(f"Visualization mode '{self.visualization_mode.name}' not implemented!")
        

    def write_detections(self, bboxes_motorcycles: torch.Tensor,cls_motorcycles: torch.Tensor, confs_motorcycles: torch.Tensor, file_name: str, bboxes_helmets: List[torch.Tensor]=None,cls_helmets: List[torch.Tensor]=None, confs_helmet: List[torch.Tensor]=None,):

        mode_path = f"Prediction_{self.mode_str}"
        full_output_path = os.path.join(self.output_path, mode_path)

        if not os.path.exists(full_output_path):
            os.makedirs(full_output_path)

        full_output_path = os.path.join(full_output_path, file_name)

        with open(full_output_path, "w") as file:

            for i in range(len(bboxes_motorcycles)):
                file.write(f"{int(cls_motorcycles[i]) if isinstance(cls_motorcycles[i], float) else cls_motorcycles[i]} "
                           f"{str(confs_motorcycles[i].item())} {str(bboxes_motorcycles[i][0].item())} "
                           f"{str(bboxes_motorcycles[i][1].item())} {str(bboxes_motorcycles[i][2].item())} "
                           f"{str(bboxes_motorcycles[i][3].item())}\n")

            if bboxes_helmets is not None:
                for j, patch in enumerate(bboxes_helmets):
                    for k in range(len(patch)):
                        if cls_helmets[j][k] == 1:
                            cur_cls_helmet = "6"  # helmet is label '6'
                        else:
                            cur_cls_helmet = "7"  # no_helmet is label '7'
                        file.write(f"{int(cur_cls_helmet)} {str(confs_helmet[j][k].item())} {str(bboxes_helmets[j][k][0].item())} {str(bboxes_helmets[j][k][1].item())} {str(bboxes_helmets[j][k][2].item())} {str(bboxes_helmets[j][k][3].item())}\n")






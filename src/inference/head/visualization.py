import os.path
import time
from enum import Enum
from typing import List, Tuple, Dict

import cv2
import numpy as np
import torch

import config
from src.inference.head.AbstractHead import AbstractHead


class VisualizationModes(Enum):
    FULL = 0  # bboxes for the motorcycles (label: #passengers) and bboxes for helmet/no_helmet
    SIMPLE = 1  # bboxes for the motorcycles (label: #passengers and #helmets)
    MINIMAL = 2  # bboxes for the motorcycles (labeled: yes/no -> traffic violation occurring?)
    TRACKING = 3  # simple with additional tracking visuals


class SourceTypes(Enum):
    INDIVIDUAL = 0
    CONTINUOUS = 1


def convert_bbox_to_z(bbox: torch.Tensor) -> Tuple[int, int]:
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns center point in the form (x,y)
    """
    w: float = bbox[2] - bbox[0]
    h: float = bbox[3] - bbox[1]
    x = int(bbox[0] + w / 2.)
    y = int(bbox[1] + h / 2.)
    return x, y


class Visualizer(AbstractHead):

    def __init__(self, visualization_mode: VisualizationModes, source_type: SourceTypes, output_path: str, show: bool):
        self.mc_colors: Tuple[Tuple[int, int, int], Tuple[int, int, int], Tuple[int, int, int], Tuple[int, int, int]] \
            = ((0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 0, 255))  # Green, Red, Blue, Purple
        self.visualization_mode = visualization_mode
        self.source_type = source_type
        self.output_path = os.path.join(config.parent_dir, output_path)
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        if source_type == SourceTypes.CONTINUOUS:
            self.output_writer: cv2.VideoWriter = None
        self.show = show
        self.frame: int = 1
        self.start_time: float = 0

        if source_type == SourceTypes.CONTINUOUS:
            self.fps: float = 0

        if visualization_mode.TRACKING:
            self.tracks: Dict[int, Tuple[List[Tuple[int, int]], Tuple[int, int, int]]] = {}

    def __draw_motorcycle_bboxes(self, image: np.ndarray, mc_bboxes: torch.Tensor, mc_scores: torch.Tensor,
                                 mc_classes: torch.Tensor, mode: VisualizationModes, helmet_count: List[int] = None,
                                 violation: torch.Tensor = None, track_ids: torch.Tensor = None) -> np.ndarray:

        # Predefine displayed text
        text: str

        for i, bbox in enumerate(mc_bboxes):
            mc_class: int = mc_classes[i]
            color = self.mc_colors[mc_class - 1]

            if mode == VisualizationModes.FULL:
                text = f"{mc_class}: {mc_scores[i]:.2f}"

            elif mode == VisualizationModes.SIMPLE:
                text = f"P: {mc_class}, H: {helmet_count[i]}"

            elif mode == VisualizationModes.MINIMAL:
                if violation[i]:
                    text = "VIOLATION"
                    color = (0, 0, 255)  # Red
                else:
                    text = "LEGAL"
                    color = (0, 255, 0)  # Green

            elif mode == VisualizationModes.TRACKING:
                try:
                    track_id = track_ids[i]
                except:
                    track_id = -1

                text = f"P: {mc_class}, H: {helmet_count[i]}, T: {track_id}"

            else:
                raise NotImplementedError(f"Visualization mode '{self.visualization_mode.name}' not implemented!")

            cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)

            cv2.putText(image, text, (int(bbox[0]), int(bbox[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return image

    def __draw_helmet_bboxes(self, image: np.ndarray, patch_coord: List[torch.Tensor], patch_scores: List[torch.Tensor],
                             patch_classes: List[torch.Tensor], mode: VisualizationModes,
                             helmet_counts: List[int] = None,
                             violation: torch.Tensor = None, track_ids: torch.Tensor = None) -> np.ndarray:
        '''
        patch_coord: list of Nx4 Tensors Containing the bboxes coordinates xywh (normalized to 1, in global coordinate frame)
        scores: list of Nx1 Tensors Containing the confidence score for each bbox
        classes: list Nx1 Tensors Containing the predicted class for each bbox
        mode: "full" / "simple" / "minimal"
        helmet_count: number of helmets in each motorcycle patch for simple mode
        violation: violation yes/no for each motorcycle patch for minimal mode
        '''

        text: str

        for i, patch in enumerate(patch_coord):
            for j, box in enumerate(patch):
                class_num = int(patch_classes[i][j].item())
                color = self.mc_colors[class_num - 1]
                if mode == VisualizationModes.FULL:
                    score = patch_scores[i][j]
                    score_str = f"{score:.2f}"
                    text = f"{class_num}: {score_str}"
                elif mode == VisualizationModes.SIMPLE:
                    text = f"riders: {class_num}, helmets: {helmet_counts[i]}"
                elif mode == VisualizationModes.MINIMAL:
                    if violation[j]:
                        text = "VIOLATION"
                        color = (0, 0, 255)  # Red
                    else:
                        text = "okay"
                        color = (0, 255, 0)  # Green
                elif mode == VisualizationModes.TRACKING:
                    text = f"r: {class_num}, h: {helmet_counts[j]}, t: {track_ids[j]}"
                else:
                    raise NotImplementedError(f"Visualization mode '{self.visualization_mode.name}' not implemented!")

                cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)

                cv2.putText(image, text, (int(box[0]), int(box[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return image

    def __visualize_tracking(self, image: np.ndarray, bboxes: torch.Tensor, classes: torch.Tensor,
                             helmets: List[int], track_ids: torch.Tensor):
        for bbox, cls, hlm, trk in zip(bboxes, classes.tolist(), helmets, track_ids.tolist()):
            if trk not in self.tracks:
                # Create entry for track with starting point and random color
                self.tracks[trk] = [convert_bbox_to_z(bbox)], np.random.random(size=3) * 256
            else:
                # Append center point to list
                self.tracks[trk][0].append(convert_bbox_to_z(bbox))

            color: Tuple[int, int, int] = self.tracks[trk][1]

            # Draw motorcycle bboxes
            text: str = f"P: {cls}, H: {hlm}, T: {trk}"
            cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.putText(image, text, (int(bbox[0]), int(bbox[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Draw track lines
            points: List[Tuple[int, int]] = self.tracks[trk][0]
            if len(points) > 1:
                # Connect points with lines
                for i in range(len(points) - 1):
                    cv2.line(image, points[i], points[i + 1], color, 2)
            else:
                # Only draw first point
                cv2.circle(image, points[0], 2, color, -1)

        return image

    def __count_fps(self, steps: int) -> float:
        elapsed_time = time.time() - self.start_time
        self.start_time = time.time()
        return steps / elapsed_time  # FPS

    def start(self) -> None:
        # Create video writer if source is video
        if self.source_type == SourceTypes.CONTINUOUS:
            fps = 10
            frame_size = (1024, 576)
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.output_writer = cv2.VideoWriter(os.path.join(self.output_path, "output.mp4"), fourcc, fps, frame_size)

    def forward(self, image_tensor: torch.Tensor, mc_bbox: torch.Tensor, mc_scores: torch.Tensor,
                mc_classes: torch.Tensor, helmet_bbox: List[torch.Tensor], helmet_scores: List[torch.Tensor],
                helmet_classes: List[torch.Tensor], track_ids: torch.Tensor = None):

        if self.source_type == SourceTypes.CONTINUOUS:
            # Update FPS every x frames
            steps = 10
            if self.frame % steps == 0:
                self.fps = self.__count_fps(steps)

        # Transform image to numpy array
        image: np.ndarray = np.transpose(image_tensor.numpy(), (1, 2, 0))

        # Determine amount of helmets
        helmet_counts: List[int] = [torch.sum(tensor == 1).item() for tensor in helmet_classes]

        # Visualize results in image depending on mode
        if self.visualization_mode == VisualizationModes.FULL:

            '''
            Full Mode means drawing the 
                1.bboxes for the motorcycles --> labeled with number of persons and confidence
                2.helmet bboxes for each motorcycle --> labeled with helmet/no-helmet and the confidence
            '''

            image = self.__draw_motorcycle_bboxes(image, mc_bbox, mc_scores, mc_classes, self.visualization_mode)
            image = self.__draw_helmet_bboxes(image, helmet_bbox, helmet_scores, helmet_classes,
                                              self.visualization_mode)

        elif self.visualization_mode == VisualizationModes.SIMPLE:

            '''
            Simple Mode means drawing the 
                1.bboxes for the motorcycles --> labeled with number of persons and number of helmets
            '''

            image = self.__draw_motorcycle_bboxes(image, mc_bbox, mc_scores, mc_classes, self.visualization_mode,
                                                  helmet_count=helmet_counts)

        elif self.visualization_mode == VisualizationModes.MINIMAL:

            '''
            Minimal Mode means drawing the 
                1.bboxes for the motorcycles --> labeled with yes/no (depending on rule violation)
            '''

            violation: torch.Tensor = torch.full((len(helmet_classes),), False, dtype=torch.bool)

            for i, patch_classes in enumerate(helmet_classes):
                violation[i] = mc_classes[i] > helmet_counts[i] or mc_classes[i] > 3

            image = self.__draw_motorcycle_bboxes(image, mc_bbox, mc_scores, mc_classes, self.visualization_mode,
                                                  violation=violation)

        elif self.visualization_mode == VisualizationModes.TRACKING:

            '''
            Simple Mode with additional tracking visuals
            '''

            image = self.__visualize_tracking(image, mc_bbox, mc_classes, helmet_counts, track_ids)

        else:
            raise NotImplementedError(f"Visualization mode '{self.visualization_mode.name}' not implemented!")

        # Draw FPS
        if self.source_type == SourceTypes.CONTINUOUS:
            cv2.putText(image, f"{self.fps:.1f}", (image.shape[1] - 55, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                        4)

        if self.show:
            cv2.imshow(f'Prediction - {self.visualization_mode.name} Mode', image)
        else:
            if self.source_type == SourceTypes.CONTINUOUS:
                # Write the frame to the video
                self.output_writer.write(image)
            else:
                save_path = os.path.join(self.output_path, f"Prediction_{self.frame:04d}.jpg")
                cv2.imwrite(save_path, image)

        # Increment frame
        self.frame += 1

        # Waits for key if source is individual images, otherwise not
        cv2.waitKey(self.source_type.value)

    def stop(self) -> None:
        # Release the VideoWriter
        self.output_writer.release()
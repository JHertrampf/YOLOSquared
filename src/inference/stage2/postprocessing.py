import copy
import torch
from ultralytics.engine.results import Results


class Postprocessor:

    def __init__(self, img_size_2):

        # compute scale for size diff. btw. total image and motorcycle patches
        self.image_size_x = img_size_2[1]
        self.image_size_y = img_size_2[0]

    def process(self, prediction: Results, bboxes_mc: torch.Tensor):

        bbox_helmet_list, conf_list, cls_list = [], [], []
        for pred in prediction:
            bbox_helmet_list.append(pred.boxes.xyxy)  # elements: [num_helmets, 4]
            conf_list.append(pred.boxes.conf)  # elements: [num_helmets, 4]
            cls_list.append(pred.boxes.cls)  # elements: [num_helmets, 4]

        bbox_helmet_list_cp = copy.deepcopy(bbox_helmet_list)

        # convert bboxes_helmet into the original image coordinate system
        mc_x_min = bboxes_mc[:, 0]
        mc_y_min = bboxes_mc[:, 1]

        for i, mc in enumerate(bbox_helmet_list_cp):
            width = bboxes_mc[i][2] - bboxes_mc[i][0]
            height = bboxes_mc[i][3] - bboxes_mc[i][1]
            scale_x = width / self.image_size_x
            scale_y = height / self.image_size_y
            for helmet in mc:
                helmet[0] = helmet[0] * scale_x + mc_x_min[i]
                helmet[1] = helmet[1] * scale_y + mc_y_min[i]
                helmet[2] = helmet[2] * scale_x + mc_x_min[i]
                helmet[3] = helmet[3] * scale_y + mc_y_min[i]

        cls_list = [tensor + 1 for tensor in cls_list]

        return bbox_helmet_list_cp, conf_list, cls_list
    
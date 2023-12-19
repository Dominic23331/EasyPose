import numpy as np
from typing import List

from .base_model import BaseModel
from .utils import letterbox, nms


class RTMDet(BaseModel):
    def __init__(self,
                 model_path: str,
                 conf_threshold: float,
                 iou_threshold: float,
                 device: str = 'CUDA',
                 warmup: int = 30):
        super(RTMDet, self).__init__(model_path, device, warmup)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.dx = 0
        self.dy = 0
        self.scale = 0

    def preprocess(self, image: np.ndarray):
        th, tw = self.input_shape[2:]
        image, self.dx, self.dy, self.scale = letterbox(image, (tw, th))
        tensor = (image - np.array((103.53, 116.28, 123.675))) / np.array((57.375, 57.12, 58.395))
        tensor = np.expand_dims(tensor, axis=0).transpose((0, 3, 1, 2)).astype(np.float32)
        return tensor

    def postprocess(self, tensor: List[np.ndarray]):
        boxes = tensor[0]
        boxes = np.squeeze(boxes, axis=0)
        boxes[..., [4, 5]] = boxes[..., [5, 4]]

        boxes = nms(boxes, self.iou_threshold, self.conf_threshold)

        boxes[:, 0] -= self.dx
        boxes[:, 2] -= self.dx
        boxes[:, 1] -= self.dy
        boxes[:, 3] -= self.dy

        boxes = np.clip(boxes, a_min=0, a_max=None)
        boxes[:, :4] /= self.scale

        return boxes

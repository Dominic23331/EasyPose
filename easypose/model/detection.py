import numpy as np
from typing import List

from .base_model import BaseModel
from .utils import letterbox, nms, xywh2xyxy


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

        human_class = boxes[..., -1] == 0
        boxes = boxes[human_class][..., :4]

        boxes[:, 0] -= self.dx
        boxes[:, 2] -= self.dx
        boxes[:, 1] -= self.dy
        boxes[:, 3] -= self.dy

        boxes = np.clip(boxes, a_min=0, a_max=None)
        boxes[:, :4] /= self.scale

        return boxes


class Yolov8(BaseModel):
    def __init__(self,
                 model_path: str,
                 conf_threshold: float,
                 iou_threshold: float,
                 device: str = 'CUDA',
                 warmup: int = 30):
        super(Yolov8, self).__init__(model_path, device, warmup)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.dx = 0
        self.dy = 0
        self.scale = 0

    def preprocess(self, image):
        th, tw = self.input_shape[2:]
        image, self.dx, self.dy, self.scale = letterbox(image, (tw, th))
        tensor = image / 255.
        tensor = np.expand_dims(tensor, axis=0).transpose((0, 3, 1, 2)).astype(np.float32)
        return tensor

    def postprocess(self, tensor):
        feature_map = tensor[0]
        feature_map = np.squeeze(feature_map, axis=0).transpose((1, 0))

        pred_class = feature_map[..., 4:]
        pred_conf = np.max(pred_class, axis=-1, keepdims=True)
        pred_class = np.argmax(pred_class, axis=-1, keepdims=True)
        boxes = np.concatenate([feature_map[..., :4], pred_conf, pred_class], axis=-1)

        boxes = xywh2xyxy(boxes)
        boxes = nms(boxes, self.iou_threshold, self.conf_threshold)

        human_class = boxes[..., -1] == 0
        boxes = boxes[human_class][..., :4]

        boxes[:, 0] -= self.dx
        boxes[:, 2] -= self.dx
        boxes[:, 1] -= self.dy
        boxes[:, 3] -= self.dy
        boxes = np.clip(boxes, a_min=0, a_max=None)
        boxes[:, :4] /= self.scale
        return boxes


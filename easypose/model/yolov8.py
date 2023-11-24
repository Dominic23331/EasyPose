import numpy as np

from .base_model import BaseModel
from .utils import letterbox, nms


class Yolov8(BaseModel):
    def __init__(self, model_path, conf_threshold, iou_threshold, device='CUDA'):
        super(Yolov8, self).__init__(model_path, device)
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
        pred_conf = np.max(pred_class, axis=-1)
        pred_class = np.argmax(pred_class, axis=-1, keepdims=True)
        boxes = np.concatenate([feature_map[..., :4], pred_conf, pred_class], axis=-1)
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

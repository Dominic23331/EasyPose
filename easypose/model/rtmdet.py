import numpy as np

from .base_model import BaseModel
from .utils import letterbox


class RTMDet(BaseModel):
    def __init__(self, model_path, device='CUDA'):
        super(RTMDet, self).__init__(model_path, device)
        self.dx = 0
        self.dy = 0
        self.scale = 0

    def preprocess(self, image):
        th, tw = self.input_shape[2:]
        image, self.dx, self.dy, self.scale = letterbox(image, (tw, th))
        tensor = (image - np.array(103.53, 116.28, 123.675)) / np.array(57.375, 57.12, 58.395)
        tensor = np.expand_dims(tensor, axis=0).transpose((0, 3, 1, 2)).astype(np.float32)
        return tensor

    def postprocess(self, tensor):
        box = tensor[0][:, :-1]
        label = tensor[0][:, :-1]
        box = box[label == 0]

        box[:, 0] -= self.dx
        box[:, 2] -= self.dx
        box[:, 1] -= self.dy
        box[:, 3] -= self.dy

        box = np.clip(box, a_min=0, a_max=None)
        box[:, :4] /= self.scale

        return box

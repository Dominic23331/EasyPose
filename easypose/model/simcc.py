import numpy as np

from .base_model import BaseModel
from .utils import letterbox, simcc_decoder


class SimCC(BaseModel):
    def __init__(self, model_path, device='CUDA'):
        super(SimCC, self).__init__(model_path, device)
        self.dx = 0
        self.dy = 0
        self.scale = 0

    def preprocess(self, image):
        th, tw = self.input_shape[2:]
        image, self.dx, self.dy, self.scale = letterbox(image, (tw, th))
        tensor = (image - np.array((103.53, 116.28, 123.675))) / np.array((57.375, 57.12, 58.395))
        tensor = np.expand_dims(tensor, axis=0).transpose((0, 3, 1, 2)).astype(np.float32)
        return tensor

    def postprocess(self, tensor):
        simcc_x, simcc_y = tensor
        simcc_x = np.squeeze(simcc_x, axis=0)
        simcc_y = np.squeeze(simcc_y, axis=0)
        keypoints = simcc_decoder(simcc_x,
                                  simcc_y,
                                  self.input_shape[2:],
                                  self.dx,
                                  self.dy,
                                  self.scale)

        return keypoints

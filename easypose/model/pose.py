import numpy as np
from typing import List

from .base_model import BaseModel
from .utils import letterbox, get_heatmap_points, \
    get_real_keypoints, refine_keypoints_dark, refine_keypoints, simcc_decoder


class Heatmap(BaseModel):
    def __init__(self,
                 model_path: str,
                 dark: bool = False,
                 device: str = 'CUDA',
                 warmup: int = 30):
        super(Heatmap, self).__init__(model_path, device, warmup)
        self.use_dark = dark
        self.img_size = ()

    def preprocess(self, image: np.ndarray):
        th, tw = self.input_shape[2:]
        self.img_size = image.shape[:2]
        image, _, _, _ = letterbox(image, (tw, th))
        tensor = (image - np.array((103.53, 116.28, 123.675))) / np.array((57.375, 57.12, 58.395))
        tensor = np.expand_dims(tensor, axis=0).transpose((0, 3, 1, 2)).astype(np.float32)
        return tensor

    def postprocess(self, tensor: List[np.ndarray]):
        heatmaps = tensor[0]
        heatmaps = np.squeeze(heatmaps, axis=0)
        keypoints = get_heatmap_points(heatmaps)
        if self.use_dark:
            keypoints = refine_keypoints_dark(keypoints, heatmaps, 11)
        else:
            keypoints = refine_keypoints(keypoints, heatmaps)
        keypoints = get_real_keypoints(keypoints, heatmaps, self.img_size)
        return keypoints


class SimCC(BaseModel):
    def __init__(self, model_path: str, device: str = 'CUDA', warmup: int = 30):
        super(SimCC, self).__init__(model_path, device, warmup)
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

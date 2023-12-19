import numpy as np
from typing import List

from .base_model import BaseModel
from .utils import letterbox, get_heatmap_points, get_real_keypoints, refine_keypoints_dark, refine_keypoints


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


import numpy as np

from .base_model import BaseModel
from .utils import letterbox, get_heatmap_points, get_real_keypoints, refine_keypoints_dark, refine_keypoints


class Heatmap(BaseModel):
    def __init__(self, model_path, dark=False, blur_kernel_size=11, device='CUDA'):
        super(Heatmap, self).__init__(model_path, device)
        self.use_dark = dark
        self.blur_kernel_size = blur_kernel_size
        self.img_size = ()

    def preprocess(self, image):
        th, tw = self.input_shape[2:]
        self.img_size = image.shape[:2]
        image, _, _, _ = letterbox(image, (tw, th))
        tensor = (image - np.array((103.53, 116.28, 123.675))) / np.array((57.375, 57.12, 58.395))
        tensor = np.expand_dims(tensor, axis=0).transpose((0, 3, 1, 2)).astype(np.float32)
        return tensor

    def postprocess(self, tensor):
        heatmaps = tensor[0]
        keypoints = get_heatmap_points(heatmaps)
        if self.use_dark:
            keypoints = refine_keypoints_dark(keypoints, heatmaps, self.blur_kernel_size)
        else:
            keypoints = refine_keypoints(keypoints, heatmaps)
        keypoints = get_real_keypoints(keypoints, heatmaps, self.img_size)
        return keypoints[0]


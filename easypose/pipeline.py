import os

import cv2
import numpy as np

import model
from .download import get_url, get_model_path, download
from .consts import AvailablePoseModels, AvailableDetModels, ROOT_PATH
from .common import Person, region_of_interest, restore_keypoints


def get_pose_model(pose_model_path, pose_model_decoder, device, warmup):
    if pose_model_decoder == 'dark':
        pose_model = model.Heatmap(pose_model_path, dark=True, device=device, warmup=warmup)
    else:
        pose_model = getattr(model, pose_model_decoder)(pose_model_path, device=device, warmup=warmup)
    return pose_model


def get_det_model(det_model_path, model_type, conf_thre, iou_thre, device, warmup):
    det_model = getattr(model, model_type)(det_model_path, conf_thre, iou_thre, device, warmup)
    return det_model


class TopDown:
    def __init__(
            self,
            pose_model_name,
            pose_model_decoder,
            det_model_name,
            conf_threshold=0.6,
            iou_threshold=0.6,
            device='cuda',
            warmup=30
    ):
        if pose_model_name not in AvailablePoseModels.POSE_MODELS:
            raise ValueError(
                'The {} human pose estimation model is not in the model repository.'.format(pose_model_name))
        if pose_model_decoder not in AvailablePoseModels.POSE_MODELS[pose_model_name]:
            raise ValueError(
                'No {} decoding head for the {} model was found in the model repository.'.format(pose_model_decoder,
                                                                                                 pose_model_name))
        if det_model_name not in AvailableDetModels.DET_MODELS:
            raise ValueError(
                'The {} detection model is not in the model repository.'.format(det_model_name))

        pose_model_path = get_model_path(AvailablePoseModels.POSE_MODELS[pose_model_name][pose_model_decoder],
                                         detection_model=False)
        if os.path.exists(pose_model_path):
            try:
                self.pose_model = get_pose_model(pose_model_path, pose_model_decoder, device, warmup)
            except:
                url = get_url(AvailablePoseModels.POSE_MODELS[pose_model_name][pose_model_decoder],
                              detection_model=False)
                download(url, pose_model_path)
                self.pose_model = get_pose_model(pose_model_path, pose_model_decoder,device, warmup)
        else:
            url = get_url(AvailablePoseModels.POSE_MODELS[pose_model_name][pose_model_decoder],
                          detection_model=False)
            download(url, pose_model_path)
            self.pose_model = get_pose_model(pose_model_path, pose_model_decoder, device, warmup)

        det_model_path = get_model_path(AvailableDetModels.DET_MODELS[det_model_name]['file_name'],
                                        detection_model=True)
        det_model_type = AvailableDetModels.DET_MODELS[det_model_name]['model_type']
        if os.path.exists(det_model_path):
            try:
                self.det_model = get_det_model(det_model_path,
                                               det_model_type,
                                               conf_threshold,
                                               iou_threshold,
                                               device,
                                               warmup)
            except:
                url = get_url(AvailableDetModels.DET_MODELS[det_model_name]['file_name'],
                              detection_model=True)
                download(url, det_model_path)
                self.det_model = get_det_model(det_model_path,
                                               det_model_type,
                                               conf_threshold,
                                               iou_threshold,
                                               device,
                                               warmup)
        else:
            url = get_url(AvailableDetModels.DET_MODELS[det_model_name]['file_name'],
                          detection_model=True)
            download(url, det_model_path)
            self.det_model = get_det_model(det_model_path,
                                           det_model_type,
                                           conf_threshold,
                                           iou_threshold,
                                           device,
                                           warmup)

    def predict(self, image):
        boxes = self.det_model(image)
        results = []
        for i in range(boxes.shape[0]):
            p = Person()
            p.box = boxes[i]
            region = region_of_interest(image, p.box)
            kp = self.pose_model(region)
            p.keypoints = restore_keypoints(p.box, kp)
            results.append(p)
        return results


class Pose:
    def __init__(self,
                 pose_model_name,
                 pose_model_decoder,
                 device='cuda',
                 warmup=30):
        if pose_model_name not in AvailablePoseModels.POSE_MODELS:
            raise ValueError(
                'The {} human pose estimation model is not in the model repository.'.format(pose_model_name))
        if pose_model_decoder not in AvailablePoseModels.POSE_MODELS[pose_model_name]:
            raise ValueError(
                'No {} decoding head for the {} model was found in the model repository.'.format(pose_model_decoder,
                                                                                                 pose_model_name))

        pose_model_path = get_model_path(AvailablePoseModels.POSE_MODELS[pose_model_name][pose_model_decoder],
                                         detection_model=False)
        if os.path.exists(pose_model_path):
            try:
                self.pose_model = get_pose_model(pose_model_path, pose_model_decoder, device, warmup)
            except:
                url = get_url(AvailablePoseModels.POSE_MODELS[pose_model_name][pose_model_decoder],
                              detection_model=False)
                download(url, pose_model_path)
                self.pose_model = get_pose_model(pose_model_path, pose_model_decoder, device, warmup)
        else:
            url = get_url(AvailablePoseModels.POSE_MODELS[pose_model_name][pose_model_decoder],
                          detection_model=False)
            download(url, pose_model_path)
            self.pose_model = get_pose_model(pose_model_path, pose_model_decoder, device, warmup)

    def predict(self, image):
        p = Person()
        box = np.array([0, 0, image.shape[3], image.shape[2], 1, 0])
        p.box = box
        p.keypoints = self.pose_model(image)
        return p

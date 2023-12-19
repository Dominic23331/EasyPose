import numpy as np
import cv2


class Person:
    def __init__(self):
        self.box = None
        self.keypoints = None


def region_of_interest(image, box):
    start_x, start_y, end_x, end_y = box
    start_x, start_y, end_x, end_y = int(start_x), int(start_y), int(end_x), int(end_y)
    cropped_image = image[start_y:end_y, start_x:end_x]
    return cropped_image


def restore_keypoints(box, keypoints):
    start_x, start_y = box[:2]
    keypoints[..., 0] += start_x
    keypoints[..., 1] += start_y
    return keypoints

import numpy as np
import cv2

from .consts import coco_skeleton


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


def draw_keypoints(image,
                   kps,
                   radius=5,
                   thickness=-1,
                   kp_color=(0, 255, 0),
                   line_color=(255, 0, 0),
                   box_color=(0, 0, 255),
                   box_thickness=5,
                   draw_box=False):
    def draw(img,
             kp,
             r,
             t,
             color,
             l_color):
        for p1, p2 in coco_skeleton:
            cv2.rectangle(img,
                          (int(kp.keypoints[p1][0]), int(kp.keypoints[p1][1])),
                          (int(kp.keypoints[p2][0]), int(kp.keypoints[p2][1])),
                          color=l_color,
                          thickness=r)
        for point in kp.keypoints:
            cv2.circle(img,
                       (int(point[0]), int(point[1])),
                       radius=r,
                       color=color,
                       thickness=t)
        return img

    if isinstance(kps, Person):
        image = draw(image, kps, radius, thickness, kp_color, line_color)
        return image
    elif isinstance(kps, list):
        for point in kps:
            if draw_box:
                cv2.rectangle(image,
                              (int(point.box[0]), int(point.box[1])),
                              (int(point.box[2]), int(point.box[3])),
                              color=box_color,
                              thickness=box_thickness)
            image = draw(image,
                         point,
                         radius,
                         thickness,
                         kp_color,
                         line_color)
        return image
    else:
        raise TypeError("Invalid format for 'kp'. Please provide a list or Person object.")





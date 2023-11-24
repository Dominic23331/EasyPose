import cv2
import numpy as np


def letterbox(img, target_size, fill_value=128):
    h, w = img.shape[:2]
    tw, th = target_size

    scale = min(tw / w, th / h)
    nw, nh = int(w * scale), int(h * scale)

    resized_img = cv2.resize(img, (nw, nh))

    canvas = np.full((th, tw, img.shape[2]), fill_value, dtype=img.dtype)

    dx, dy = (tw - nw) // 2, (th - nh) // 2
    canvas[dy:dy + nh, dx:dx + nw, :] = resized_img

    return canvas, dx, dy, scale


def intersection_over_union(box1, box2):
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = (x2 - x1) * (y2 - y1)
    union = area1 + area2 - intersection
    iou = intersection / (union + 1e-6)

    return iou


def nms(boxes, iou_threshold, conf_threshold):
    conf = boxes[..., 4] > conf_threshold
    boxes = boxes[conf]
    boxes = list(boxes)
    boxes.sort(reverse=True, key=lambda x: x[4])

    result = []
    while boxes:
        chosen_box = boxes.pop()

        b = []
        for box in boxes:
            if box[-1] != chosen_box[-1] or \
               intersection_over_union(chosen_box, box) \
               < iou_threshold:
                b.append(box)

        result.append(chosen_box)
        boxes = b

    return np.array(result)

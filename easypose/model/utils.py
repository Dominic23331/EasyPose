from itertools import product
from typing import Sequence

import cv2
import numpy as np


def letterbox(img: np.ndarray, target_size: Sequence[int], fill_value: int = 128):
    h, w = img.shape[:2]
    tw, th = target_size

    scale = min(tw / w, th / h)
    nw, nh = int(w * scale), int(h * scale)

    resized_img = cv2.resize(img, (nw, nh))

    canvas = np.full((th, tw, img.shape[2]), fill_value, dtype=img.dtype)

    dx, dy = (tw - nw) // 2, (th - nh) // 2
    canvas[dy:dy + nh, dx:dx + nw, :] = resized_img

    return canvas, dx, dy, scale


def intersection_over_union(box1: np.ndarray, box2: np.ndarray):
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


def nms(boxes: np.ndarray, iou_threshold: float, conf_threshold: float):
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


def get_heatmap_points(heatmap: np.ndarray):
    keypoints = np.zeros([1, heatmap.shape[0], 3], dtype=np.float32)
    for i in range(heatmap.shape[0]):
        h, w = np.where(heatmap[i] == heatmap[i].max())
        h, w = h[0], w[0]
        h_fixed = h + 0.5
        w_fixed = w + 0.5
        score = heatmap[i][h][w]
        keypoints[0][i][0] = w_fixed
        keypoints[0][i][1] = h_fixed
        keypoints[0][i][2] = score
    return keypoints


def gaussian_blur(heatmaps: np.ndarray, kernel: int = 11):
    assert kernel % 2 == 1

    border = (kernel - 1) // 2
    K, H, W = heatmaps.shape

    for k in range(K):
        origin_max = np.max(heatmaps[k])
        dr = np.zeros((H + 2 * border, W + 2 * border), dtype=np.float32)
        dr[border:-border, border:-border] = heatmaps[k].copy()
        dr = cv2.GaussianBlur(dr, (kernel, kernel), 0)
        heatmaps[k] = dr[border:-border, border:-border].copy()
        heatmaps[k] *= origin_max / np.max(heatmaps[k])
    return heatmaps


def refine_keypoints(keypoints: np.ndarray, heatmaps: np.ndarray):
    N, K = keypoints.shape[:2]
    H, W = heatmaps.shape[:2]

    for n, k in product(range(N), range(K)):
        x, y = keypoints[n, k, :2].astype(int)

        if 1 < x < W - 1 and 0 < y < H:
            dx = heatmaps[k, y, x + 1] - heatmaps[k, y, x - 1]
        else:
            dx = 0.

        if 1 < y < H - 1 and 0 < x < W:
            dy = heatmaps[k, y + 1, x] - heatmaps[k, y - 1, x]
        else:
            dy = 0.

        keypoints[n, k] += np.sign([dx, dy], dtype=np.float32) * 0.25

    return keypoints


def refine_keypoints_dark(keypoints: np.ndarray, heatmaps: np.ndarray, blur_kernel_size: int = 11):
    N, K = keypoints.shape[:2]
    H, W = heatmaps.shape[1:]

    # modulate heatmaps
    heatmaps = gaussian_blur(heatmaps, blur_kernel_size)
    np.maximum(heatmaps, 1e-10, heatmaps)
    np.log(heatmaps, heatmaps)

    for n, k in product(range(N), range(K)):
        x, y = keypoints[n, k, :2].astype(int)
        if 1 < x < W - 2 and 1 < y < H - 2:
            dx = 0.5 * (heatmaps[k, y, x + 1] - heatmaps[k, y, x - 1])
            dy = 0.5 * (heatmaps[k, y + 1, x] - heatmaps[k, y - 1, x])

            dxx = 0.25 * (
                    heatmaps[k, y, x + 2] - 2 * heatmaps[k, y, x] +
                    heatmaps[k, y, x - 2])
            dxy = 0.25 * (
                    heatmaps[k, y + 1, x + 1] - heatmaps[k, y - 1, x + 1] -
                    heatmaps[k, y + 1, x - 1] + heatmaps[k, y - 1, x - 1])
            dyy = 0.25 * (
                    heatmaps[k, y + 2, x] - 2 * heatmaps[k, y, x] +
                    heatmaps[k, y - 2, x])
            derivative = np.array([[dx], [dy]])
            hessian = np.array([[dxx, dxy], [dxy, dyy]])
            if dxx * dyy - dxy ** 2 != 0:
                hessianinv = np.linalg.inv(hessian)
                offset = -hessianinv @ derivative
                offset = np.squeeze(np.array(offset.T), axis=0)
                keypoints[n, k, :2] += offset
    return keypoints


def get_real_keypoints(keypoints: np.ndarray, heatmaps: np.ndarray, img_size: Sequence[int]):
    img_h, img_w = img_size
    heatmap_h, heatmap_w = heatmaps.shape[1:]
    heatmap_ratio = heatmaps.shape[1] / heatmaps.shape[2]
    img_ratio = img_h / img_w
    if heatmap_ratio > img_ratio:
        resize_w = img_w
        resize_h = int(img_w * heatmap_ratio)
    elif heatmap_ratio < img_ratio:
        resize_h = img_h
        resize_w = int(img_h / heatmap_ratio)
    else:
        resize_w = img_w
        resize_h = img_h

    keypoints[:, :, 0] = (keypoints[:, :, 0] / heatmap_w) * resize_w - (resize_w - img_w) / 2
    keypoints[:, :, 1] = (keypoints[:, :, 1] / heatmap_h) * resize_h - (resize_h - img_h) / 2

    keypoints = np.squeeze(keypoints, axis=0)

    return keypoints


def simcc_decoder(simcc_x: np.ndarray,
                  simcc_y: np.ndarray,
                  input_size: Sequence[int],
                  dx: int,
                  dy: int,
                  scale: float):
    x = np.argmax(simcc_x, axis=-1, keepdims=True).astype(np.float32)
    y = np.argmax(simcc_y, axis=-1, keepdims=True).astype(np.float32)

    x_conf = np.max(simcc_x, axis=-1, keepdims=True)
    y_conf = np.max(simcc_y, axis=-1, keepdims=True)
    conf = (x_conf + y_conf) / 2

    x /= simcc_x.shape[-1]
    y /= simcc_y.shape[-1]
    x *= input_size[1]
    y *= input_size[0]

    keypoints = np.concatenate([x, y, conf], axis=-1)
    keypoints[..., 0] -= dx
    keypoints[..., 1] -= dy
    keypoints[..., :2] /= scale

    return keypoints

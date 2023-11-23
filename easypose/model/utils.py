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

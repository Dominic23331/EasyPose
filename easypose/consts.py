import os
from collections import OrderedDict

from .__version__ import __version__

home = os.path.expanduser("~")
ROOT_PATH = os.path.join(home, ".easypose")
ROOT_URL = "https://huggingface.co/dominic23331/easypose/resolve/main"
VERSION = __version__


class AvailablePoseModels(object):
    POSE_MODELS = OrderedDict(
        {
            'hrnet': {
                'Heatmap': 'hrnet_heatmap_coco_256x192_20240115.onnx',
                'Dark': 'hrnet_dark_coco_256x192_20240115.onnx',
            },
            'litehrnet': {
                'Heatmap': 'litehrnet_heatmap_coco_256x192_20231009.onnx',
            },
            'resnet50': {
                'Heatmap': 'resnet_heatmap_coco_256x192_20231009.onnx',
                'SimCC': 'resnet_simcc_coco_256x192_20231009.onnx',
            },
            'rtmpose-tiny': {
                'SimCC': 'rtmpose_t_simcc_coco_256x192_20231009.onnx',
            },
            'rtmpose-s': {
                'SimCC': 'rtmpose_s_simcc_coco_256x192_20231009.onnx',
            }
        }
    )


class AvailableDetModels(object):
    DET_MODELS = OrderedDict(
        {
            'rtmdet_s': {
                'file_name': 'rtmdet_s_coco_640x640_20231209.onnx',
                'model_type': 'RTMDet',
            },
            'rtmdet_tiny': {
                'file_name': 'rtmdet_tiny_coco_640x640_20231209.onnx',
                'model_type': 'RTMDet',
            },
            'yolov8_n': {
                'file_name': 'yolov8_n_coco_640x640_20231124.onnx',
                'model_type': 'Yolov8',
            },
            'yolov8_s': {
                'file_name': 'yolov8_s_coco_640x640_20231124.onnx',
                'model_type': 'Yolov8',
            }
        }
    )


def print_list(lst):
    if not lst:
        print("Input list is empty.")
        return

    # 打印列表头
    print("Index | Value")
    print("-" * 15)

    # 打印列表元素
    for index, value in enumerate(lst):
        print(f"{index:<6} | {value}")


def print_dict(input_dict):
    if not input_dict:
        print("Input dictionary is empty.")
        return

    # 获取最长键的长度
    max_key_length = max(len(str(key)) for key in input_dict.keys())

    # 打印表头
    print(f"{'Key':<{max_key_length}} | Value")
    print("-" * (max_key_length + 8))  # 8 是为了留出空格

    # 打印键值对
    for key, value in input_dict.items():
        print(f"{str(key):<{max_key_length}} | {value}")


def det_model_list():
    det_models = list(AvailableDetModels.DET_MODELS.keys())
    print_list(det_models)

def pose_model_list():
    pose_model = {}
    models = list(AvailablePoseModels.POSE_MODELS.keys())
    for i in range(len(models)):
        pose_model[models[i]] = list(AvailablePoseModels.POSE_MODELS[models[i]].keys())
    print_dict(pose_model)


coco_skeleton = [
    [0, 1], [1, 3], [0, 2], [2, 4], [5, 6], [5, 7],
    [7, 9], [6, 8], [8, 10], [5, 11], [6, 12], [11, 12],
    [11, 13], [13, 15], [12, 14], [14, 16]
]

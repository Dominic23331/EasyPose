from collections import OrderedDict

from __version__ import __version__


ROOT_PATH = "~/.easypose"
ROOT_URL = "https://huggingface.co/dominic23331/easypose/resolve/main"
VERSION = __version__


class AvailablePoseModels(object):
    POSE_MODELS = OrderedDict(
        {
            'litehrnet_w18': {
                'heatmap': {
                    'file_name': 'litehrnet_w18_heatmap_coco_256x192_20231009.onnx'
                }
            },
            'litehrnet_w30': {
                'heatmap': {
                    'file_name': 'litehrnet_w30_heatmap_coco_256x192_20231009.onnx'
                }
            },
            'resnet50': {
                'heatmap': {
                    'file_name': 'resnet_heatmap_coco_256x192_20231009.onnx'
                },
                'simcc': {
                    'file_name': 'resnet_simcc_coco_256x192_20231009.onnx'
                }
            },
            'rtmpose-tiny': {
                'simcc': {
                    'file_name': 'rtmpose_tiny_simcc_coco_256x192_20231009.onnx'
                }
            },
            'rtmpose-s': {
                'simcc': {
                    'file_name': 'rtmpose_s_simcc_coco_256x192_20231009.onnx'
                }
            }
        }
    )


class AvailableDetModels(object):
    DET_MODELS = OrderedDict(
        {
            'rtmdet': {
                's': {
                    'file_name': 'rtmdet_s_coco_640x640_20231123.onnx',
                    'type': 'rtmdet'
                },
                'tiny': {
                    'file_name': 'rtmdet_tiny_coco_640x640_20231123.onnx',
                    'type': 'rtmdet'
                }
            }
        }
    )

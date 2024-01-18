# EasyPose

## Introduction

[EasyPose](https://github.com/Dominic23331/EasyPose)is a human pose estimation algorithm library based on Python and Onnxruntime, which integrates commonly used human pose estimation algorithms and can be used after installation. The original intention of developing EasyPose is to enable developers to easily use various human pose estimation algorithms for their own tasks. Therefore, EasyPose has fewer third-party dependencies and is more convenient to use. When developing a human pose estimation application using EasyPose, the program automatically downloads the corresponding weight file from the server and only requires less than ten lines of code to obtain the algorithm's prediction results.

<details open>
<summary>Major features</summary>



- **Simple operation**

  EasyPose can quickly call various human pose estimation algorithms with simple commands and supports custom models, greatly facilitating developers to quickly use algorithms.

- **Supports multiple models**

  EasyPose supports various human pose estimation models, including HRNet and RTMPose, as well as human detection models such as RTMDet and YOLOv8.

- **Fast speed**

  The model implements the GPU version and can quickly call algorithms within the GPU.

</details>

------

## Install

1、Install anaconda and create a new virtual environment

```bash
conda create -n easypose python=3.8
```

2、Clone the repository and install EasyPose

```
git clone https://github.com/Dominic23331/EasyPose.git
pip install -v -e .
```

3、Verify installation

```
import easypose as ep
print(ep.pose_model_list())
```

------

## Models

### Human detection algorithm

| Model       | Input Size |  AP  | PARAMS | GFLOPS |
| :---------- | :--------: | :--: | :----: | ------ |
| RTMDet-tiny |  640x640   | 41.1 |  4.8   | 8.1    |
| RTMDet-s    |  640x640   | 44.6 |  8.89  | 14.8   |
| YOLOv8-n    |  640x640   | 37.3 |  3.2   | 8.7    |
| YOLOv8-s    |  640x640   | 44.9 |  11.2  | 28.6   |

The above model is taken from [mmdetection](https://github.com/open-mmlab/mmdetection) and [ultralytics](https://github.com/ultralytics/ultralytics).

### Human pose estimation algorithm

| Model               | Input Size | AP   | AR   |
| ------------------- | ---------- | ---- | ---- |
| RTMPose-tiny        | 256x192    | 68.2 | 73.6 |
| RTMPose-s           | 256x192    | 71.6 | 76.8 |
| RTMPose-m           | 256x192    | 74.6 | 79.5 |
| RTMPose-l           | 256x192    | 75.8 | 80.6 |
| ResNet50-SimCC      | 256x192    | 72.1 | 78.1 |
| ResNet50-Heatmap    | 256x192    | 72.0 | 77.5 |
| HRNet-Heatmap       | 256x192    | 74.9 | 80.4 |
| HRNet-Dark          | 256x192    | 75.7 | 80.7 |
| Hourglass           | 256x192    | 72.6 | 78.0 |
| Lite-HRNet-Heatmap  | 256x192    | 64.2 | 70.5 |
| MobileNetv2-Heatmap | 256x192    | 64.8 | 70.9 |
| MobileNetv2-SimCC   | 256x192    | 62.0 | 67.8 |

The above model is taken from [mmpose](https://github.com/open-mmlab/mmpose)。

------

## Getting Start

1、Importing the EasyPose and OpenCV libraries

```python
import easypose as ep
import cv2
```

2、Instantiating the TopDown model

```python
model = ep.TopDown('rtmpose_s', 'SimCC', 'rtmdet_s')
```

3、Using the predict function to predict input images

```python
image = cv2.imread('img.jpg')
poses = model.predict(image)
```

4、Draw key points of the human body in the image

```python
image = ep.draw_keypoints(image, poses)
```

------

## Open Source License

EasyPose follows [Apache 2.0](https://github.com/Dominic23331/EasyPose/blob/master/LICENSE) open source license.

------

## Subsequent tasks

- [ ] Add more TopDown human pose estimation algorithms
- [ ] Add some one-stage human pose estimation algorithms
- [ ] Optimize the speed of existing models and add quantitative models
- [ ] Writing instructional documents
- [ ] Publish wheel files in pypi
- [ ] Support whole body pose estimation algorithms
- [ ] Support MPII dataset
- [ ] Suppoty animal pose estimation algorithms
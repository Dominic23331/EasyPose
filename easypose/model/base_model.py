import warnings
from abc import ABC, abstractmethod
from typing import List

import numpy as np
import onnxruntime as ort


class BaseModel(ABC):
    def __init__(self, model_path: str, device: str = 'CUDA', warmup: int = 30):
        self.opt = ort.SessionOptions()

        if device == 'CUDA':
            provider = 'CUDAExecutionProvider'
            if provider not in ort.get_available_providers():
                warnings.warn("No CUDAExecutionProvider found, switched to CPUExecutionProvider.", UserWarning)
                provider = 'CPUExecutionProvider'
        elif device == 'CPU':
            provider = 'CPUExecutionProvider'
        else:
            raise ValueError('Provider {} does not exist.'.format(device))

        self.session = ort.InferenceSession(model_path,
                                            providers=[provider],
                                            sess_options=self.opt)

        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape

        if warmup > 0:
            self.warmup(warmup)

    @abstractmethod
    def preprocess(self, image: np.ndarray):
        pass

    @abstractmethod
    def postprocess(self, tensor: List[np.ndarray]):
        pass

    def forward(self, image: np.ndarray):
        tensor = self.preprocess(image)
        result = self.session.run(None, {self.input_name: tensor})
        output = self.postprocess(result)
        return output

    def warmup(self, epoch: int = 30):
        tensor = np.random.random(self.input_shape)
        for i in range(epoch):
            self.session.run(None, {self.input_name: tensor})

    def __call__(self, image: np.ndarray, *args, **kwargs):
        return self.forward(image)

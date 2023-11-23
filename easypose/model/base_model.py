from abc import ABC, abstractmethod

import onnxruntime as ort


class BaseModel(ABC):
    def __init__(self, model_path, device='CUDA'):
        self.opt = ort.SessionOptions()

        if device == 'CUDA':
            provider = 'CUDAExecutionProvider'
        elif device == 'CPU':
            provider = 'CPUExecutionProvider'
        else:
            raise ValueError('Provider {} does not exist.'.format(device))

        self.session = ort.InferenceSession(model_path, providers=[provider])

        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape

    @abstractmethod
    def preprocess(self, image):
        pass

    @abstractmethod
    def postprocess(self, tensor):
        pass

    def forward(self, image):
        tensor = self.preprocess(image)
        result = self.session.run(None, {self.input_name: tensor})
        output = self.postprocess(result)
        return output

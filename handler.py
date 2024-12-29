import torch
import os
import tempfile
from PIL import Image
from torchvision import transforms
import numpy as np
from ts.torch_handler.base_handler import BaseHandler

class ModelHandler(BaseHandler):
    img_size = 640
    threshold = 0.5
    """
    A custom model handler implementation.
    """

    def __init__(self):
        super().__init__()
        self.mapping = [
            'heavy_truck', 
            'medium_truck', 
            'light_truck', 
            'large_bus', 
            'minibus', 
            'microbus', 
            'utility', 
            'car/taxi', 
            'auto_rickshaw', 
            'tempo', 
            'motorcycle', 
            'bicycle', 
            'cycle_rickshaw', 
            'rickshaw_van', 
            'animal/pushcart'
            ]
        self.manifest = None
        self.model = None
        self.initialized = False

    def __bbox_cxcywh_to_xyxy(self, boxes):
        half_width = boxes[:, 2] / 2
        half_height = boxes[:, 3] / 2
        
        # Calculate the coordinates of the bounding boxes
        x_min = boxes[:, 0] - half_width
        y_min = boxes[:, 1] - half_height
        x_max = boxes[:, 0] + half_width
        y_max = boxes[:, 1] + half_height
        
        # Return the bounding boxes in (x_min, y_min, x_max, y_max) format
        return np.column_stack((x_min, y_min, x_max, y_max))

    def initialize(self, context):
        """
        Invoke by torchserve for loading a model
        :param context: context contains model server system properties
        :return:
        """

        #  load the model
        self.manifest = context.manifest

        properties = context.system_properties
        model_dir = properties.get("model_dir")
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")

        # Read model serialize/pt file
        serialized_file = self.manifest['model']['serializedFile']
        model_pt_path = os.path.join(model_dir, serialized_file)
        if not os.path.isfile(model_pt_path):
            raise RuntimeError("Missing the model.pt file")
        
        self.model = torch.jit.load(model_pt_path)

        self.initialized = True

    def preprocess(self, data):
        """
        Transform raw input into model input data.
        :param batch: list of raw requests, should match batch size
        :return: list of preprocessed model input data
        """
        images = []

        # handle if images are given in base64, etc.
        if isinstance(data, list):
            data = data[0]
        data = data.get("data") or data.get("body")

        # print('55-------', type(data)) # should be bytearray

        fp = tempfile.NamedTemporaryFile()
        fp.write(data)
        fp.seek(0)
        image = Image.open(fp.name)
        self.image_width, self.image_height = image.size
        transform = transforms.Compose(
            [
                transforms.Resize((ModelHandler.img_size, ModelHandler.img_size)), 
                transforms.ToTensor()
            ]
        )
        img = transform(image)
        image_tensor = torch.tensor(img.clone().detach().to(self.device)).unsqueeze(0)
        print(f"Input Image shape: {image_tensor.shape}")
        return image_tensor
    
    def postprocess(self, inference_output):
        """
        Return inference result.
        :param inference_output: list of inference output
        :return: list of predict results
        """
        # Take output from network and post-process to desired format
        outputs = inference_output[0].squeeze(0).t()
        boxes = outputs[:, :4]
        scores = outputs[:, 4:]
        classes = torch.argmax(scores, dim = 1)
        scores = torch.max(scores, dim = 1).values
        mask = scores > ModelHandler.threshold
        boxes = boxes[mask]
        boxes = self.__bbox_cxcywh_to_xyxy(boxes)
        classes = classes[mask]
        scores = scores[mask]

        boxes[:, 0::2] *= self.image_width / ModelHandler.img_size
        boxes[:, 1::2] *= self.image_height / ModelHandler.img_size

        detections = []
        for cls, box, score in zip(classes, boxes, scores):
            detections.append(
                {
                    'class': self.mapping[int(cls)],
                    'box': box.tolist(),
                    'conf': score.tolist()
                }
            )
        print(detections)

        return [detections]

    def handle(self, data, context):
        """
        Invoke by TorchServe for prediction request.
        Do pre-processing of data, prediction using model and postprocessing of prediciton output
        :param data: Input data for prediction
        :param context: Initial context contains model server system properties.
        :return: prediction output
        """
        data = self.preprocess(data)
        with torch.no_grad():
            results = self.model(data)
        return self.postprocess(results)
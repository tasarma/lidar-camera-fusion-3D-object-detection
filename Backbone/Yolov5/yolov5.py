import cv2
import torch
import numpy as np

class YoloV5(torch.nn.Module):
    def __init__(self):
        super(YoloV5, self).__init__()
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        self.features = torch.nn.Sequential(*list(self.model.children())[:-1])
    
    def process_image(self, img: np.ndarray) -> np.ndarray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 640))
        img = img.transpose(2, 0, 1)  # Convert to channels first format
        img = img.astype('float32') / 255.0
        img = torch.from_numpy(img).unsqueeze(0)  # Add batch dimension

        return img

    def forward(self, x):
        # self.process_image(x)
        x = self.features(x)
        return x

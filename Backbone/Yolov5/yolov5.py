import cv2
import torch
import numpy as np
import torchvision.models as models

class YoloV5(torch.nn.Module):
    def __init__(self):
        super(YoloV5, self).__init__()
        self.model = self.__load_module()
        self.features = torch.nn.Sequential(*list(self.model.children())[:-1])
    
    def process_image(self, img: np.ndarray) -> np.ndarray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 640))
        img = img.transpose(2, 0, 1)  # Convert to channels first format
        img = img.astype('float32') / 255.0
        img = torch.from_numpy(img).unsqueeze(0)  # Add batch dimension

        return img
    
    def __load_module(self):
        try:
            return torch.hub.load('ultralytics/yolov5', 'yolov5s')
        except Exception as e:
            print(f"Failed to download 'yolov5s' module: {e}")
            print("Loading pre-trained 'yolov5s.pt' file...")
            return torch.hub.load('ultralytics/yolov5', 'custom', path='./yolov5s.pt')

    def forward(self, x, batch_size):
        # self.process_image(x)
        x = self.features(x)
        print(type(x), x.shape, x.size)
        x = x.view(batch_size, 1, 2048)
        return x


class ResNet50(torch.nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        self.res50_model = models.resnet50(pretrained=True)
        self.features = torch.nn.Sequential(*list(self.res50_model.children())[:-1])
        
    def forward(self, x, batch_size):
        # self.process_image(x)
        print('buradaaaaa ', type(x), x.shape, x.size)
        x = x.permute(0, 3, 1, 2)
        x = self.features(x)
        x = x.view(batch_size, 1, 2048)
        return x
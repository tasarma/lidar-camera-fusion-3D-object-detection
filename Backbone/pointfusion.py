import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision.transforms

from .Pointnet.pointnet import PointNetEncoder
from .Yolov5.yolov5 import YoloV5, ResNet50


class Fusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.pointnet = PointNetEncoder()
        # self.yolov5 = YoloV5()
        self.resnet = ResNet50()

        self.fc1 = nn.Linear(3136, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 8*3) # 8*3 for 3D bounding box coordinates
        self.fc5 = nn.Linear(128, 1) 

    def forward(self, img, pts):
        batch_size = img.size()[0]
        B, D, N = pts.size() # Batch size, number of points, number of channels

        # base_feat = self.yolov5(img, batch_size)
        base_feat = self.resnet(img, batch_size)
        # print(base_feat.shape)
        global_feat, point_feat= self.pointnet(pts)

        base_feat = F.normalize(base_feat, p=2, dim=2)
        global_feat = F.normalize(global_feat, p=2, dim=2)
        point_feat = F.normalize(point_feat, p=2, dim=2)

        base_feat = base_feat.repeat(1, D, 1)
        global_feat = global_feat.repeat(1, D, 1)
        
        # fusion
        # print(base_feat.shape, global_feat.shape, point_feat.shape)
        fusion_feat = torch.cat((base_feat, global_feat, point_feat), dim=2)  # 180
        print('fusion shape' ,fusion_feat.shape)
        
        fusion_feat = self.fc1(fusion_feat)
        fusion_feat = F.relu(self.fc2(fusion_feat))
        fusion_feat = F.relu(self.fc3(fusion_feat))

        corner_offsets = self.fc4(fusion_feat)
        corner_offsets = corner_offsets.view(batch_size, D, 8, 3)
        
        scores = self.fc5(fusion_feat)
        scores = scores.view(-1, D)
        
        # Shift scores so minimum is 0
        minimum = (scores.min(dim=1)[0]).view(batch_size,-1)
        scores = scores - minimum

        # Add eps to prevent returning 0 
        eps = 1e-4
        scores = scores + eps

        # Divide by range to normalize 
        s_range = scores.max(dim = 1)[0] - scores.min(dim=1)[0]
        s_range = s_range.view(batch_size,-1)
        scores = scores/s_range 
        
        return corner_offsets, scores

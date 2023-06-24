from typing import List, Union
import argparse

import yaml
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from Utils.visualization import draw_lidar_bbox, show_image_with_boxes, display_lidar
from Backbone.pointfusion import Fusion
from DataProcess.kitti_dataset import KittiDataset

# path = r'/media/dara/699c1837-e7a1-4428-8624-493535df96c5/KITTI/dataset/kitti'
path = r'/home/dara/Workspace/Bitirme/Dataset/kitti'
data = KittiDataset(root=path, mode='train') 

img, obj, calib, lidar = data.__getitem__(2)
points = lidar[:, :3]
# img, obj, calib, lidar = data.__g `etitem__(142)
# show_image_with_boxes(img, obj, calib, show3d=True)
# draw_lidar_bbox(points, obj, calib)
display_lidar(points)
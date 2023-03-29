import cv2
import torch
import open3d
import numpy as np

from DataProcess.kitti_dataset import KittiDataset
from Utils import visualization as vis

root = r'/home/tasarma/Playground/Tez/dataset/kitti'
data = KittiDataset(root=root, mode='train')

def getCorners(points):
    # get the corners from the transformed pointcloud
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(points)
    obb = open3d.geometry.OrientedBoundingBox()
    obb = obb.create_from_points(pcd.points)
    corners = np.asarray(obb.get_box_points())
    return corners

# to able to use this function 
# first you need to change dtype of numpy array to "float64"
# and change size of the array: from [... , 4] (x, y, z, reflactnece) to [... , 3] (x, y, z)
# then you can use this function


def get_corner_offsets(corners: np.ndarray, cloud: np.ndarray) -> np.ndarray:
    cnt = cloud.shape[0]
    corner_offsets = cloud.reshape(cnt, 1, 3) - corners
    return corner_offsets # (cnt, 8, 3)

# data.__getitem__(6)

# points, corners = data.__getitem__(3)

# pts = np.squeeze(points)
# pts = pts.astype(np.float64)
# crs = np.squeeze(corners)
# crs = crs.astype(np.float64)
# of = get_corner_offsets(crs, pts)

# vis.show_lidar_with_boxes(pts, crs)



from Backbone.Yolov5.yolov5 import YOLOv5
model = YOLOv5()

img = data.get_image(3)

feat = model(img[..., ::-1])  # OpenCV image (BGR to RGB)
print(feat.shape)
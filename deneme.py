import cv2
import torch
import open3d
import numpy as np
from PIL import Image

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

roi_imgs, roi_pcs, obj_list = data.__getitem__(13)

# def simg(img):
#     cv2.imshow('img', img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# vis.display_lidar(pts_lidar[:, :3])
# vis.display_lidar(ret_pts)
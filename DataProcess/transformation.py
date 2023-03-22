import os
import sys
import numpy as np
from typing import List

import _init_path
from config.kitti_config import R0_inv, Tr_velo_to_cam_inv


def inverse_rigid_trans(Tr: List[float]) -> np.ndarray:
    ''' Inverse a rigid body transform matrix (3x4 as [R|t])
        [R'|-R't; 0|1]
    '''
    inv_Tr = np.zeros_like(Tr)  # 3x4
    inv_Tr[0:3, 0:3] = np.transpose(Tr[0:3, 0:3])
    inv_Tr[0:3, 3] = np.dot(-np.transpose(Tr[0:3, 0:3]), Tr[0:3, 3])
    return inv_Tr


def camera_to_lidar(x: float, y: float, z: float, 
                    V2C=None, R0=None, P2=None) -> tuple[np.ndarray]:
    p = np.array([x, y, z, 1])
    if V2C is None or R0 is None:
        p = np.matmul(R0_inv, p) # config. BURAYI DUZELT
        p = np.matmul(Tr_velo_to_cam_inv, p) # config. BURAYI DUZELT
    else:
        R0_i = np.zeros((4, 4))
        R0_i[:3, :3] = R0
        R0_i[3, 3] = 1
        p = np.matmul(np.linalg.inv(R0_i), p)
        p = np.matmul(inverse_rigid_trans(V2C), p)
    p = p[0:3]
    return tuple(p)


def camera_to_lidar_box(boxes: np.ndarray, 
                        V2C=None, R0=None, P2=None) -> np.ndarray:
    # (N, 7) -> (N, 7) x,y,z,h,w,l,r
    ret = []
    for box in boxes:
        x, y, z, h, w, l, ry = box
        (x, y, z), h, w, l, rz = camera_to_lidar(
            x, y, z, V2C=V2C, R0=R0, P2=P2), h, w, l, -ry - np.pi / 2
        # rz = angle_in_limit(rz)
        ret.append([x, y, z, h, w, l, rz])
    return np.array(ret).reshape(-1, 7)



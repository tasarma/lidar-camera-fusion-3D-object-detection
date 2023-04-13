from typing import List, Dict, Tuple

import cv2
import scipy
import numpy as np
from PIL import Image
from scipy.spatial import Delaunay

from . import transformation

class Object3D:
    """
    3D Object Label. For more info:
    https://vasavi-kosaraju02-research.medium.com/exploration-of-kitti-dataset-for-autonomous-driving-6dd7c34f7eae
    """
    def __init__(
            self, 
            line_in_label_file: str
            ):
        data = line_in_label_file.split(' ')
        data[1:] = [float(x) for x in data[1:]]

        # extract label, truncation, occlusion
        self.cls_type = data[0]  # 'Car', 'Pedestrian', ...
        self.cls_id = self.cls_type_to_id(self.cls_type)
        self.truncation = data[1]  # truncated pixel ratio [0..1]
        self.occlusion = int(data[2])  # 0=visible, 1=partly occluded, 2=fully occluded, 3=unknown
        self.alpha = data[3]  # object observation angle [-pi..pi]

        # extract 2d bounding box in 0-based coordinates
        self.xmin = data[4]  # left
        self.ymin = data[5]  # top
        self.xmax = data[6]  # right
        self.ymax = data[7]  # bottom
        self.box2d = np.array([self.xmin, self.ymin, self.xmax, self.ymax])

        # extract 3d bounding box information
        self.h = data[8]  # box height
        self.w = data[9]  # box width
        self.l = data[10]  # box length (in meters)
        self.t = (data[11], data[12], data[13])  # location (x,y,z) in camera coord.
        self.dis_to_cam = np.linalg.norm(self.t)
        self.ry = data[14]  # yaw angle (around Y-axis in camera coordinates) [-pi..pi]
        self.score = data[15] if data.__len__() == 16 else -1.0
        self.level_str = None
        self.level = self.get_obj_level()

    def cls_type_to_id(self, cls_type: str) -> int:
        CLASS_NAME_TO_ID = {
            'Car': 0,
            'Pedestrian': 1,
            'Cyclist': 2,
            'Van': 3
        }
        if cls_type not in CLASS_NAME_TO_ID.keys():
            return -1
        return CLASS_NAME_TO_ID[cls_type]
  
    def get_obj_level(self) -> int:
        height = float(self.box2d[3]) - float(self.box2d[1]) + 1

        if height >= 40 and self.truncation <= 0.15 and self.occlusion <= 0:
            self.level_str = 'Easy'
            return 1  # Easy
        elif height >= 25 and self.truncation <= 0.3 and self.occlusion <= 1:
            self.level_str = 'Moderate'
            return 2  # Moderate
        elif height >= 25 and self.truncation <= 0.5 and self.occlusion <= 2:
            self.level_str = 'Hard'
            return 3  # Hard
        else:
            self.level_str = 'UnKnown'
            return 4

    def print_object(self) -> str:
        print(f'Type: {self.cls_type}, truncation: {self.truncation} \
              occlusion: {self.occlusion}, alpha: {self.alpha}')
        print(f'2d bbox (x0,y0,x1,y1): \
              {self.xmin, self.ymin, self.xmax, self.ymax}')
        print(f'3d bbox h, w, l: {self.h, self.w, self.l}')
        print(f'3d bbox location, ry: {self.t[0], self.t[1], self.t[2], self.ry}')

    def to_kitti_format(self) -> str:
        kitti_str = f'{self.cls_type}, {self.truncation:.2f}, {int(self.occlusion)},\
                      {self.alpha:.2f}, {self.box2d[0]:.2f}, {self.box2d[1]:.2f}, \
                      {self.box2d[2]:.2f}, {self.box2d[3]:.2f}, {self.h:.2f}, \
                      {self.w:.2f}, {self.l:.2f}, {self.t[0]:.2f}, {self.t[1]:.2f},\
                      {self.t[2]:.2f}, {self.ry:.2f}, {self.score:.2f}'

        return kitti_str


def read_label(label_file_path: str) -> List[Object3D]:
    with open(label_file_path) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
        objects = [Object3D(line) for line in lines]
        return objects


class Calibration:
    def __init__(
            self,
            calib_file_path: str
    ):
        calibs = self.read_calib_file(calib_file_path)
        # Projection matrix from rect camera coord to image2 coord
        self.P2 = calibs['P2']

        # Rigid transform from Velodyne coord to reference camera coord
        self.V2C = calibs['Tr_velo_to_cam']
        self.C2V = transformation.inverse_rigid_trans(self.V2C)

        # Rotation from reference camera coord to rect camera coord
        self.R0 = calibs['R0']

    def read_calib_file(self, file_path: str) -> Dict[str, np.ndarray]:
        with open(file_path) as f:
            lines = f.readlines()
    
        obj = lines[2].strip().split(' ')[1:]
        P2 = np.array(obj, dtype=np.float32)
        obj = lines[4].strip().split(' ')[1:]
        R0 = np.array(obj, dtype=np.float32)
        obj = lines[5].strip().split(' ')[1:]
        Tr_velo_to_cam = np.array(obj, dtype=np.float32)
    
        return {'P2': P2.reshape(3, 4),
                'R0': R0.reshape(3, 3),
                'Tr_velo_to_cam': Tr_velo_to_cam.reshape(3, 4)}    
          
    def cart_to_hom(self, pts: np.ndarray) -> np.ndarray:
        """:param pts: (N, 3 or 2)
        :return pts_hom: (N, 4 or 3)
        """
        pts_hom = np.hstack((pts, np.ones((pts.shape[0], 1), dtype=np.float32)))
        return pts_hom
  
    # ===========================
    # ------- 3d to 3d ----------
    # ===========================
    def project_velo_to_ref(self, pts_3d_velo: np.ndarray) -> np.ndarray:
        pts_3d_velo = self.cart_to_hom(pts_3d_velo)  # nx4
        return np.dot(pts_3d_velo, np.transpose(self.V2C))
    
    def project_ref_to_velo(self, pts_3d_ref):
        pts_3d_ref = self.cart_to_hom(pts_3d_ref)  # nx4
        return np.dot(pts_3d_ref, np.transpose(self.C2V))

    def project_rect_to_ref(self, pts_3d_rect: np.ndarray) -> np.ndarray:
        """Input and Output are nx3 points """
        return np.transpose(np.dot(np.linalg.inv(self.R0), 
                                   np.transpose(pts_3d_rect)))
    
    def project_rect_to_velo(self, pts_3d_rect):
        ''' Input: nx3 points in rect camera coord.
            Output: nx3 points in velodyne coord.
        '''
        pts_3d_ref = self.project_rect_to_ref(pts_3d_rect)
        return self.project_ref_to_velo(pts_3d_ref)
  

    def project_ref_to_rect(self, pts_3d_ref: np.ndarray) -> np.ndarray:
        """ Input and Output are nx3 points """
        return np.transpose(np.dot(self.R0, np.transpose(pts_3d_ref)))

    def project_velo_to_rect(self, pts_3d_velo: np.ndarray) -> np.ndarray:
        pts_3d_ref = self.project_velo_to_ref(pts_3d_velo)
        return self.project_ref_to_rect(pts_3d_ref)

    # ===========================
    # ------- 3d to 2d ----------
    # ===========================
    def project_rect_to_image(self, pts_3d_rect: np.ndarray) -> List[np.ndarray]:
        """ Input: nx3 points in rect camera coord.
        Output: nx3 points in image2 coord.
        """
        pts_3d_rect = self.cart_to_hom(pts_3d_rect)
        pts_2d = np.dot(pts_3d_rect, self.P2.T)  # nx3
        pts_2d[:, 0] /= pts_2d[:, 2]
        pts_2d[:, 1] /= pts_2d[:, 2]
        pts_2d[:, 2] -= self.P2.T[3, 2] # depth in rect cam coordinates
        return pts_2d

    def project_velo_to_image(self, pts_3d_velo: np.ndarray) -> np.ndarray:
        """ Input: nx3 points in velodyne coord.
        Output: nx2 points in image2 coord.
        """
        pts_3d_rect = self.project_velo_to_rect(pts_3d_velo)
        pts_2d = self.project_rect_to_image(pts_3d_rect)
        pts_img = pts_2d[:, 0:2] # x and y coordinates
        pts_rect_depth = pts_2d[:, 2] # z coordinate
        return (pts_3d_rect, pts_img, pts_rect_depth)


def check_labels(objects) -> Tuple[np.ndarray, bool]:
    bbox_selected = []
    for obj in objects:
        if obj.cls_id != -1:
            bbox = []
            bbox.append(obj.cls_id)
            bbox.extend([obj.t[0], obj.t[1], obj.t[2], obj.h, obj.w, obj.l, obj.ry])
            bbox_selected.append(bbox)

    if len(bbox_selected) == 0:
        labels = np.zeros((1, 8), dtype=np.float32)
        noObjectLabels = True
    else:
        labels = np.array(bbox_selected, dtype=np.float32)
        noObjectLabels = False

    return labels, noObjectLabels


def get_boxes3d(obj_list: list) -> np.ndarray:
    """Extracts the 3D box information"""
    box3d_list = np.zeros((obj_list.__len__(), 7), dtype=np.float32)
    for i, obj in enumerate(obj_list):
        box3d_list[i, 0:3] = obj.t # location (x,y,z) in camera coord.
        box3d_list[i, 3] = obj.h # box height
        box3d_list[i, 4] = obj.w # box width
        box3d_list[i, 5] = obj.l # box length (in meters)
        box3d_list[i, 6] = obj.ry # yaw angle 

    return box3d_list


def roty(t):
    # Rotation about the y-axis.
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])


def box3d_to_corner3d(boxes3d: np.ndarray, rotate: bool = True) -> np.ndarray:
    """
    param boxes3d: (N, 7) [x, y, z, h, w, l, ry]
    return: corners3d: (N, 8, 3)
    """
    boxes_num = boxes3d.shape[0]

    # 3d bounding box dimensions
    l = boxes3d[:, 5]
    w = boxes3d[:, 4]
    h = boxes3d[:, 3]

    # 3d bounding box corners (N, 8)
    x_corners = np.array([l / 2., l / 2., -l / 2., -l / 2., l / 2., l / 2., 
                          -l / 2., -l / 2.], dtype=np.float32).T 

    z_corners = np.array([w / 2., -w / 2., -w / 2., w / 2., w / 2., -w / 2., 
                          -w / 2., w / 2.], dtype=np.float32).T 

    y_corners = np.zeros((boxes_num, 8), dtype=np.float32) 
    y_corners[:, 4:8] = -h.reshape(boxes_num, 1).repeat(4, axis=1)

    # compute rotational matrix around yaw axis
    if rotate:
        ry = boxes3d[:, 6]
        zeros = np.zeros(ry.size, dtype=np.float32)
        ones = np.ones(ry.size, dtype=np.float32)

        rot_list = np.array([[np.cos(ry), zeros, -np.sin(ry)],
                             [zeros,       ones,       zeros],
                             [np.sin(ry), zeros,  np.cos(ry)]])  # (3, 3, N)

        R_list = np.transpose(rot_list, (2, 0, 1))  # (N, 3, 3)

        temp_corners = np.concatenate((x_corners.reshape(-1, 8, 1), 
                                       y_corners.reshape(-1, 8, 1),
                                       z_corners.reshape(-1, 8, 1)), axis=2)  # (N, 8, 3)

        rotated_corners = np.matmul(temp_corners, R_list)  # (N, 8, 3)

        x_corners = rotated_corners[:, :, 0]
        y_corners = rotated_corners[:, :, 1] 
        z_corners = rotated_corners[:, :, 2]

    x_loc, y_loc, z_loc = boxes3d[:, 0], boxes3d[:, 1], boxes3d[:, 2]

    x = x_loc.reshape(-1, 1) + x_corners.reshape(-1, 8)
    y = y_loc.reshape(-1, 1) + y_corners.reshape(-1, 8)
    z = z_loc.reshape(-1, 1) + z_corners.reshape(-1, 8)

    corners = np.concatenate((x.reshape(-1, 8, 1), 
                              y.reshape(-1, 8, 1), 
                              z.reshape(-1, 8, 1)), axis=2)

    return corners.astype(np.float32)

def enlarge_box3d(boxes3d, extra_width):
    """
    boxes3d: (N, 7) [x, y, z, h, w, l, ry]
    """
    if isinstance(boxes3d, np.ndarray):
        large_boxes3d = boxes3d.copy()
    else:
        large_boxes3d = boxes3d.clone()

    large_boxes3d[:, 3:6] += extra_width * 2
    large_boxes3d[:, 1] += extra_width

    return large_boxes3d

def in_hull(p, hull):
    """
    p: (N, K) test points
    hull: (M, K) M corners of a box
    return (N) bool
    """
    try:
        if not isinstance(hull, Delaunay):
            hull = Delaunay(hull)
        flag = hull.find_simplex(p) >= 0

    except scipy.spatial.qhull.QhullError:
        print(f'Warning: not a hull {hull}')
        flag = np.zeros(p.shape[0], dtype=np.bool)

    return flag

def compute_box_3d(obj, P):
    """ Takes an object and a projection matrix (P) and projects the 3d
        bounding box into the image plane.
        Returns:
            corners_2d: (8,2) array in left image coord.
            corners_3d: (8,3) array in in rect camera coord.
    """
    # compute rotational matrix around yaw axis
    R = roty(obj.ry)

    # 3d bounding box dimensions
    l = obj.l
    w = obj.w
    h = obj.h

    # 3d bounding box corners
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

    # rotate and translate 3d bounding box
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))

    corners_3d[0, :] = corners_3d[0, :] + obj.t[0]
    corners_3d[1, :] = corners_3d[1, :] + obj.t[1]
    corners_3d[2, :] = corners_3d[2, :] + obj.t[2]

    # only draw 3d bounding box for objs in front of the camera
    if np.any(corners_3d[2, :] < 0.1):
        corners_2d = None
        return corners_2d, np.transpose(corners_3d)

    # project the 3d bounding box into the image plane
    corners_2d = transformation.project_to_image(np.transpose(corners_3d), P)

    return corners_2d, np.transpose(corners_3d)

def crop_image(img: np.ndarray, obj: Object3D, width: int=640) -> np.ndarray:
    l, t, r, b = obj.box2d.astype(int)
    scale_box = 0
    roi = img[t+scale_box:b+scale_box, l+scale_box:r+scale_box]
    roi = cv2.resize(roi, (width, width), interpolation=cv2.INTER_AREA)

    return roi

def crop_lidar(points: np.ndarray, obj: Object3D, calib: Calibration) -> np.ndarray:
    """ Extract the points inside the bbox"""
    box3d_pts_2d, box3d_pts_3d = compute_box_3d(obj, calib.P2)
    box3d_pts_velo = calib.project_rect_to_velo(box3d_pts_3d)

    mask = np.logical_and.reduce((
        points[:, 0] >= box3d_pts_velo[:, 0].min(), points[:, 0] <= box3d_pts_velo[:, 0].max(),
        points[:, 1] >= box3d_pts_velo[:, 1].min(), points[:, 1] <= box3d_pts_velo[:, 1].max(),
        points[:, 2] >= box3d_pts_velo[:, 2].min(), points[:, 2] <= box3d_pts_velo[:, 2].max()
        ))
    
    cropped_pts = points[mask]
    print('mask  ', mask.shape)
    print('cropped  ',cropped_pts.shape)

    return cropped_pts





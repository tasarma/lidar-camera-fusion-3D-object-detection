import os
import sys
from typing import List

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

import kitti_utils
from transformation import camera_to_lidar_box
from kitti_utils import Object3D
from config.kitti_config import CLASS_NAME_TO_ID



class KittiDataset(Dataset):
    def __init__(
            self,
            root: str,
            mode: str = 'train',
            num_samples: int = None
            ):
        assert mode in ['train', 'test'] , f'Invalid Mode: {mode}'
        self.mode = mode
        self.npoints = 16384
        self.classes = ['Car']
        self.use_intensity = False
        is_test = self.mode == 'test'
        self.data_dir = root
        sub_folder = 'testing' if is_test else 'training'

        self.lidar_dir = os.path.join(self.data_dir, sub_folder, "velodyne")
        self.image_dir = os.path.join(self.data_dir, sub_folder, "image_2")
        self.calib_dir = os.path.join(self.data_dir, sub_folder, "calib")
        self.label_dir = os.path.join(self.data_dir, sub_folder, "label_2")

        split_txt_path = os.path.join(self.data_dir, 'ImageSets', f'{mode}.txt')
        self.image_idx_list = [x.strip() for x in open(split_txt_path).readlines()]

        if is_test:
            self.sample_id_list = [int(sample_id) for sample_id in self.image_idx_list]
        else:
            self.sample_id_list = self.remove_invalid_idx(self.image_idx_list)

        if num_samples is not None:
            self.sample_id_list = self.sample_id_list[:num_samples]
        self.num_samples = len(self.sample_id_list)

    def __len__(self) -> int:
        return len(self.sample_id_list)

    def get_image(self, index: int) -> np.ndarray :
        image_file = os.path.join(self.image_dir, f'{index:06}.png')
        assert os.path.exists(image_file), 'File not exist'
        return cv2.imread(image_file) # (H, W, C) -> (H, W, 3) in BGR mode

    def get_lidar(self, index: int) -> np.ndarray:
        lidar_file = os.path.join(self.lidar_dir, f'{index:06}.bin')
        assert os.path.exists(lidar_file), 'File not exist'
        return np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)

    def get_calib(self, index: int) -> kitti_utils.Calibration:
        calib_file = os.path.join(self.calib_dir, f'{index:06}.txt')
        assert os.path.exists(calib_file), 'File not exist'
        return kitti_utils.Calibration(calib_file)

    def get_label(self, index: int) -> List[Object3D] :
        label_file = os.path.join(self.label_dir, f'{index:06}.txt')
        assert os.path.exists(label_file), f'File not exist {label_file}'
        return kitti_utils.read_label(label_file)

    def pad_matrix(self, matrix, fixed_size):
        """Pad the matrix to the specified fixed size"""
        n = matrix.shape[0]
        padded_matrix = np.zeros((fixed_size, matrix.shape[1]))
        padded_matrix[:n] = matrix
        return padded_matrix
        
    def __getitem__(self, index: int):
        """Get the item at the specified index in the dataset"""
        sample_id = int(self.sample_id_list[index])
        calib = self.get_calib(sample_id)
        img_shape = (self.get_image(sample_id)).shape
        pts_lidar= self.get_lidar(sample_id)

        pts_3d_rect, pts_img, pts_rect_depth = calib.project_velo_to_image(pts_lidar[:, 0:3])
        pts_intensity = pts_lidar[:, 3]
        
        # Get valid points (projected points should be in image)
        pts_valid_flag = self.get_valid_flag(pts_img, pts_rect_depth, img_shape)
        
        pts_3d_rect = pts_3d_rect[pts_valid_flag][:, 0:3]
        pts_intensity = pts_intensity[pts_valid_flag]

        choice = self.__seperate_points__(pts_3d_rect)
        ret_pts = pts_3d_rect[choice, :]
        ret_pts_intensity = pts_intensity[choice] - 0.5 # translate intensity to [-0.5, 0.5]

        pts_features = [ret_pts_intensity.reshape(-1, 1)]
        if pts_features.__len__() > 1:
            ret_pts_features = np.concatenate(pts_features, axis=1)
        else:
            ret_pts_features = pts_features[0]

        sample_info = {'sample_id': sample_id}

        if self.mode == 'test':
            if self.use_intensity:
                pts_input = np.concatenate((ret_pts, ret_pts_features), axis=1) # (n, c)
            else:
                pts_input = ret_pts

            sample_info['pts_input'] = pts_input
            sample_info['pts_rect'] = ret_pts
            sample_info['pts_features'] = ret_pts_features
            return sample_info

        obj_list = self.filtrate_objects(self.get_label(sample_id))
        box3d_list = kitti_utils.get_boxes3d(obj_list)

        # Prepare input
        if self.use_intensity:
            pts_input = np.concatenate((ret_pts, ret_pts_features), axis=1) # (n, c)
        else:
            pts_input = ret_pts

        # Generate training labels
        cls_labels = self.generate_training_labels(ret_pts, box3d_list)

        sample_info['pts_input'] = pts_input
        sample_info['pts_rect'] = ret_pts
        sample_info['cls_labels'] = cls_labels
        
        return sample_info

    @staticmethod
    def generate_training_labels(
            pts_ret: np.ndarray, 
            box3d_list: np.ndarray,
            ) -> np.ndarray:
        cls_label = np.zeros((pts_ret.shape[0]), dtype=np.int32)

        corners = kitti_utils.box3d_to_corner3d(box3d_list, rotate=True)
        extend_boxes3d = kitti_utils.enlarge_box3d(box3d_list, extra_width=0.2)
        extend_corners = kitti_utils.box3d_to_corner3d(extend_boxes3d, rotate=True)

        for k in range(box3d_list.shape[0]):
            box_corners = corners[k]
            fg_pt_flag = kitti_utils.in_hull(pts_ret, box_corners)
            cls_label[fg_pt_flag] = 1

            # enlarge the bbox3d, ignore nearby points
            extend_box_corners = extend_corners[k]
            fg_enlarge_flag = kitti_utils.in_hull(pts_ret, extend_box_corners)
            ignore_flag = np.logical_xor(fg_pt_flag, fg_enlarge_flag)
            cls_label[ignore_flag] = -1

        return cls_label

    def filtrate_objects(self, obj_list: np.ndarray) -> list:
        type_whitelist = self.classes
        if self.mode == 'TRAIN':
            type_whitelist = list(self.classes)
            if 'Car' in self.classes:
                type_whitelist.append('Van')
        valid_obj_list = []
        for obj in obj_list:
            if obj.cls_type not in type_whitelist:
                continue
            valid_obj_list.append(obj)

        return valid_obj_list

    def __seperate_points__(self, pts_3d_rect: np.ndarray) -> np.ndarray:
        """Select points according to self.npoints"""

        if self.npoints < len(pts_3d_rect):
            pts_depth = pts_3d_rect[:, 2]
            pts_near_flag = pts_depth < 40.0

            # Separate the points less than 40 and greater than 40
            far_idxs_choice = np.where(pts_near_flag == 0)[0]
            near_idxs = np.where(pts_near_flag == 1)[0]
            near_idxs_choice = np.random.choice(near_idxs, 
                                                self.npoints - len(far_idxs_choice), 
                                                replace=False)
            if len(far_idxs_choice) > 0:
                choice = np.concatenate((near_idxs_choice, far_idxs_choice), axis=0)
            else:
                choice = near_idxs_choice

        else:
            # To complete pts_3d_rect to self.npoints, add more points
            choice = np.arange(0, len(pts_3d_rect), dtype=np.int32)
            if self.npoints > len(pts_3d_rect):
                extra_choice = np.random.choice(choice, self.npoints - len(pts_3d_rect), replace=False)
                choice = np.concatenate((choice, extra_choice), axis=0)
        
        np.random.shuffle(choice)
        return choice

    @staticmethod
    def get_valid_flag(
        pts_img: np.ndarray, 
        pts_rect_depth: np.ndarray,
        img_shape: np.ndarray
    ) -> np.ndarray:
        """Filter out points which are not visible in the image"""
        val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
        val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
        val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
        pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)
        return pts_valid_flag
       
    def remove_invalid_idx(self, image_idx_list: List[str]) -> List[int]:
        """Discard samples which don't have current training class objects,
        which will not be used for training.
        """
        sample_id_list = []
        for sample_id in image_idx_list:
            sample_id = int(sample_id)
            objects = self.get_label(sample_id)
            calib = self.get_calib(sample_id)
            labels, noObjectLabels = kitti_utils.check_labels(objects)
            if not noObjectLabels:
                labels[:, 1:] = camera_to_lidar_box(labels[:, 1:],
                                                    calib.V2C, calib.R0,
                                                    calib.P2)  # convert rect cam to velo cord
            valid_list = []
            for i in range(labels.shape[0]):
                if int(labels[i, 0]) in CLASS_NAME_TO_ID.values():  #config.CLASS_NAME_TO_ID.values(): BURAYI DUZELT
                    valid_list.append(labels[i, 0])

            if len(valid_list) > 0:
                sample_id_list.append(sample_id)

        return sample_id_list

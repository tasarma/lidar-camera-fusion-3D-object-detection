import os
import sys
from typing import List, Tuple
from PIL.Image import new

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

import DataProcess.init_path as init_path
import DataProcess.kitti_utils as kitti_utils
from DataProcess.transformation import camera_to_lidar_box
from DataProcess.kitti_utils import Object3D, Calibration as Calib
from Config.kitti_config import CLASS_NAME_TO_ID
from Utils import visualization as vis


class KittiDataset(Dataset):
    def __init__(
            self,
            root: str,
            mode: str = 'train',
            num_samples: int = None
            ):
        assert mode in ['train', 'test'] , f'Invalid Mode: {mode}'
        self.mode = mode
        self.npoints = 400
        self.classes = ['Car']
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


    def __getitem__(self, index: int):
        ''' Get the item at the specified index in the dataset'''
        sample_id = int(self.sample_id_list[index])
        calib = self.get_calib(sample_id)
        img = self.get_image(sample_id)
        pts_lidar= self.get_lidar(sample_id)

        # Get valid points (projected points should be in image)
        ret_pts = self.get_lidar_in_image_fov(pts_lidar[:, :3], calib, 0, 0, img.shape[1], img.shape[0])

        sample_info = {'sample_id': sample_id}
        sample_info['roi_img'] = []
        sample_info['roi_pc'] = []

        obj_list = self.filtrate_objects(self.get_label(sample_id))

        # if len(obj_list) <= 0:
        #     sample_info['roi_img'] = None
        #     sample_info['roi_pc'] = None
        #     sample_info['class'] = None
        #     return sample_info

        for i, obj in enumerate(obj_list):
            roi_img = kitti_utils.crop_image(img, obj)
            roi_pc = kitti_utils.crop_lidar(ret_pts, obj, calib)
            # cv2.imwrite(f'cropped_{i}.jpg', roi_img)

            # Apply a mask to select points in the point cloud
            if len(roi_pc) >= 200:
                mask = self.__seperate_points(roi_pc)
                roi_pc = roi_pc[mask, :]
                # vis.display_lidar(roi_pc)
                # print(i, sample_id, roi_pc.shape)

                sample_info['roi_img'].append(roi_img)
                sample_info['roi_pc'].append(roi_pc)

        return sample_info

    def filtrate_objects(self, obj_list: np.ndarray) -> list:
        type_whitelist = self.classes
        if self.mode == 'train':
            type_whitelist = list(self.classes)
            if 'Car' in self.classes:
                type_whitelist.append('Van')
        valid_obj_list = []
        for obj in obj_list:
            if obj.cls_type not in type_whitelist:
                continue
            valid_obj_list.append(obj)

        return valid_obj_list

    def __seperate_points(self, pts_3d_rect: np.ndarray) -> np.ndarray:
        """Select points according to self.npoints"""
        len_pts = len(pts_3d_rect)
        if self.npoints < len_pts:
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
            choice = np.arange(0, len_pts, dtype=np.int32)
            if self.npoints > len_pts:
                if len(choice) >= (self.npoints - len_pts):
                    extra_choice = np.random.choice(choice, self.npoints - len_pts, replace=False)
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
                if int(labels[i, 0]) in CLASS_NAME_TO_ID.values():
                    valid_list.append(labels[i, 0])

            if len(valid_list) > 0:
                sample_id_list.append(sample_id)

        return sample_id_list

    def get_lidar_in_image_fov(self, pc_velo, calib: Calib, xmin, ymin, xmax, ymax,
                           clip_distance=0.0):
        ''' Filter lidar points, keep those in image FOV '''
        pts_3d_rect, pts_2d, _ = calib.project_velo_to_image(pc_velo)

        x_in_fov = np.logical_and(pts_2d[:, 0] < xmax, pts_2d[:, 0] >= xmin)
        y_in_fov = np.logical_and(pts_2d[:, 1] < ymax, pts_2d[:, 1] >= ymin)
        in_fov = np.logical_and(x_in_fov, y_in_fov)

        clip_mask = pc_velo[:, 0] > clip_distance
        fov_inds = np.logical_and(in_fov, clip_mask)
        imgfov_pc_velo = pc_velo[fov_inds, :]
        
        return imgfov_pc_velo

    def collate_fn(self, batch):
        batch_size = batch.__len__()
        ans_dict = {}

        for i in range(batch_size):
            for j, key in enumerate(batch[i].keys()):
                if isinstance(batch[i][key], list):
                    key_size = len(batch[i][key])
                    if key not in ans_dict:
                        ans_dict[key] = np.concatenate([batch[i][key][k][np.newaxis, ...] for k in range(key_size)], axis=0)
                    else:
                        ans_dict[key] = np.concatenate([ans_dict[key], 
                                                        np.concatenate([batch[i][key][k][np.newaxis, ...] 
                                                                        for k in range(key_size)], axis=0)], 
                                                                        axis=0)
                else:
                    if key not in ans_dict:
                        ans_dict[key] = [batch[k][key] for k in range(batch_size)]
                        if isinstance(batch[0][key], int):
                            ans_dict[key] = np.array(ans_dict[key], dtype=np.int32)

        return ans_dict


import sys

import numpy
from torchvision import transforms
from torch.utils.data import DataLoader

from DataProcess.kitti_dataset import KittiDataset


# predata processing
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def create_train_dataloader(config):
    """Create dataloader for training"""
    root = r'/home/tasarma/Playground/Tez/dataset/kitti'
    batch_size = 4

    train_dataset = KittiDataset(root=root)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader

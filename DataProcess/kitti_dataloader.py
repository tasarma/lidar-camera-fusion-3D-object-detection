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
    """
    Create dataloader for training.

    Args:
        config (dict): Configuration dictionary.

    Returns:
        DataLoader: Training dataloader.
    """
    root = config['dataset']['root_dir']
    batch_size = config['train']['batch_size']

    train_dataset = KittiDataset(root=root)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader

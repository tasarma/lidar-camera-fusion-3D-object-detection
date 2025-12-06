import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class STN3d(nn.Module):
    """
    Spatial Transformer Network for 3D point clouds.
    """
    def __init__(self, channel=3):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.iden = torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32)).reshape(1, 9)

    def forward(self, x):
        """
        Forward pass of STN3d.

        Args:
            x (torch.Tensor): Input tensor [B, 3, N].

        Returns:
            torch.Tensor: Transformation matrix [B, 3, 3].
        """
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = self.iden.repeat(batchsize, 1).to(x.device)
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    """
    Spatial Transformer Network for k-dimensional features.
    """
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k
        self.iden = torch.from_numpy(np.eye(self.k).flatten().astype(np.float32)).reshape(1, self.k * self.k)

    def forward(self, x):
        """
        Forward pass of STNkd.

        Args:
            x (torch.Tensor): Input tensor [B, k, N].

        Returns:
            torch.Tensor: Transformation matrix [B, k, k].
        """
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = self.iden.repeat(batchsize, 1).to(x.device)

        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetEncoder(nn.Module):
    """Encoder for PointNet"""

    def __init__(self,
                 in_channels: int=3,
                 input_transform: bool=True,
                 feature_transform: bool=True,
                 global_feature: bool=True,
                 is_seg: bool=False,  
                 **kwargs
                 ):
        """
        Args:
            in_channels (int): feature size of input. Defaults to 3
            input_transform (bool, optional): whether to use transformation for coordinates. Defaults to True.
            feature_transform (bool, optional): whether to use transformation for features. Defaults to True.
            global_feature (bool, optional): whether to use global for features. Defaults to True.
            is_seg (bool, optional): for segmentation or classification. Defaults to False.
        """
        super().__init__()
        self.stn = STN3d(in_channels) if input_transform else None
        self.conv0_1 = torch.nn.Conv1d(in_channels, 64, 1)
        self.conv0_2 = torch.nn.Conv1d(64, 64, 1)

        self.conv1 = torch.nn.Conv1d(64, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)

        self.bn0_1 = nn.BatchNorm1d(64)
        self.bn0_2 = nn.BatchNorm1d(64)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        
        self.fstn = STNkd(k=64) if feature_transform else None
        self.out_channels = 1024 + 64 if is_seg else 1024 

        self.global_feat = global_feature
         
    def forward(self, pos, x=None):
        """
        Forward pass of PointNetEncoder.

        Args:
            pos (torch.Tensor or dict): Input point cloud [B, 3, N] or dictionary containing 'x'.
            x (torch.Tensor, optional): Optional input features. Defaults to None.

        Returns:
            tuple:
                global_feat (torch.Tensor): Global features [B, 1, 1024].
                point_feat (torch.Tensor): Point features [B, 400, 64].
        """
        if hasattr(pos, 'keys'):
            x = pos['x']
        if x is None:
            x = pos.transpose(1, 2).contiguous()
        
        B, D, N = x.size()
        if self.stn is not None:
            trans = self.stn(x)
            x = x.transpose(2, 1)
            if D > 3:
                feature = x[:, :, 3:]
                x = x[:, :, :3]
            x = torch.bmm(x, trans)
            if D > 3:
                x = torch.cat([x, feature], dim=2)
            x = x.transpose(2, 1)
        x = F.relu(self.bn0_1(self.conv0_1(x)))
        x = F.relu(self.bn0_2(self.conv0_2(x)))

        if self.fstn is not None:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        x = F.relu(self.bn1(self.conv1(x)))

        point_feat = x
        point_feat = point_feat.view(-1, 400, 64)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]

        x = x.view(-1, 1, 1024) # global features
        
        if self.global_feat:
            return x, point_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, N)
            return point_feat.view(-1,64), trans_feat




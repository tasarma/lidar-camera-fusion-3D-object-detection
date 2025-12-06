from typing import List, Union
import argparse

import os
import yaml
import wandb
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from Utils.visualization import visualize_result, show_image_with_boxes
from Backbone.pointfusion import Fusion
from DataProcess.kitti_dataset import KittiDataset


parser = argparse.ArgumentParser(description="Arg parser")
parser.add_argument("--mode", type=str, default='train')


args = parser.parse_args()


def log_print(info: str, log_f: Union[str, None]=None) -> None:
    """
    Print logs to console and optionally to a file.

    Args:
        info (str): The message to log.
        log_f (str, optional): Path to the log file. Defaults to None.
    """
    print(info)
    if log_f is not None:
        print(info, file=log_f)


def unsupervisedLoss(pred_offsets, pred_scores, gt_pts_offsets):
    """
    Calculate the unsupervised loss for the model.

    Args:
        pred_offsets (torch.Tensor): Predicted offsets [B, pnts, 8, 3].
        pred_scores (torch.Tensor): Predicted scores [B, pnts].
        gt_pts_offsets (torch.Tensor): Ground truth point offsets [B, pnts, 8, 3].

    Returns:
        torch.Tensor: Calculated loss.
    """
    eps = 1e-16
    weight = 0.1
    L1 = nn.SmoothL1Loss(reduction='none')
    loss_offset = L1(pred_offsets, gt_pts_offsets) # [B x pnts x 8 x 3]
    loss_offset = torch.mean(loss_offset, dim=(2, 3)) # [B x pnts]
    loss = ((loss_offset * pred_scores) - (weight * torch.log(pred_scores + eps)))
    # loss = loss.mean() # [1]
    return loss

def saveCheckpoint(model, epoch, optimizer, loss, path):
    """
    Save the model checkpoint.

    Args:
        model (torch.nn.Module): The model to save.
        epoch (int): Current epoch number.
        optimizer (torch.optim.Optimizer): Optimizer state.
        loss (float): Current loss value.
        path (str): Path to save the checkpoint.
    """
    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
                }, path)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_one_epoch(
        model: Fusion, 
        train_loader: DataLoader, 
        optimizer: torch.optim.Adam, 
        epoch: int, 
        cfg: dict
        ):
    """
    Train the model for one epoch.

    Args:
        model (Fusion): The model to train.
        train_loader (DataLoader): DataLoader for training data.
        optimizer (torch.optim.Adam): Optimizer for updating model weights.
        epoch (int): Current epoch number.
        cfg (dict): Configuration dictionary.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    log_print(f'===============TRAIN EPOCH {epoch+1}================')
    running_loss = 0.
    last_loss = 0.
    
    for itr, batch in enumerate(train_loader):
        img, points = batch['roi_img'], batch['roi_pc']
        gt_pts_offsets, gt_corners = batch['gt_pts_offsets'], batch['gt_corners']
        
        img = torch.from_numpy(img).float().cuda(non_blocking=True).float()
        points = torch.from_numpy(points).float().cuda(non_blocking=True).float()
        gt_pts_offsets = torch.from_numpy(gt_pts_offsets).float().cuda(non_blocking=True).float()
        gt_corners = torch.from_numpy(gt_corners).float().cuda(non_blocking=True).float()

        optimizer.zero_grad()

        pred_offset, scores = model(img, points)
        
		# Unsupervised loss
        loss = 0
        loss = unsupervisedLoss(pred_offset, scores, gt_pts_offsets)
        loss = loss.sum(dim=1) / 400 # Number of points
        loss = loss.sum(dim=0) / cfg['train']['batch_size']

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        # Finding anchor point and predicted offset based on maximum score
        max_inds = scores.max(dim=1)[1].cpu().numpy()
        p_offset = np.zeros((4, 8, 3))
        anchor_points = np.zeros((4, 3))
        truth_boxes = np.zeros((4, 8, 3))
        for i in range(0, 4):
            p_offset[i] = pred_offset[i][max_inds[i]].cpu().detach().numpy()
            anchor_points[i] = points[i][max_inds[i]].cpu().numpy()
            truth_boxes[i] = gt_corners[i].cpu().numpy()
        
        # show_image_with_boxes(img[0], anchor_points, truth_boxes)
        # visualize_result(anchor_points, p_offset, truth_boxes)
        # loss_epoch = running_loss / cfg['train']['batch_size']
        if itr % 10 == 0 and itr != 0:
            last_loss = running_loss / 10 # loss per batch
            print(f"Epoch [{epoch}/{cfg['train']['num_epochs']}], Step [{itr}] Loss: {last_loss:.4f}")
            if cfg['train']['wandb']['use']:
                wandb.log({"train_loss": last_loss, "epoch": epoch, "step": itr + epoch * len(train_loader)})
            # saveCheckpoint(model, epoch, optimizer, loss, f'models/pointfusion_{itr}.pth')
            running_loss = 0
    
    print(f"Loss for Epoch {epoch+1} is {last_loss} and running_loss is {running_loss}")
    print(f"Loss for Epoch {epoch+1} is {last_loss} and running_loss is {running_loss}")
    return last_loss


def validate_one_epoch(
        model: Fusion, 
        val_loader: DataLoader, 
        epoch: int, 
        cfg: dict
        ):
    """
    Validate the model for one epoch.

    Args:
        model (Fusion): The model to validate.
        val_loader (DataLoader): DataLoader for validation data.
        epoch (int): Current epoch number.
        cfg (dict): Configuration dictionary.

    Returns:
        float: Average validation loss.
    """
    model.eval()
    log_print(f'===============VALIDATION EPOCH {epoch+1}================')
    running_loss = 0.
    
    with torch.no_grad():
        for itr, batch in enumerate(val_loader):
            img, points = batch['roi_img'], batch['roi_pc']
            gt_pts_offsets, gt_corners = batch['gt_pts_offsets'], batch['gt_corners']
            
            img = torch.from_numpy(img).float().cuda(non_blocking=True).float()
            points = torch.from_numpy(points).float().cuda(non_blocking=True).float()
            gt_pts_offsets = torch.from_numpy(gt_pts_offsets).float().cuda(non_blocking=True).float()
            
            pred_offset, scores = model(img, points)
            
            # Unsupervised loss
            loss = unsupervisedLoss(pred_offset, scores, gt_pts_offsets)
            loss = loss.sum(dim=1) / 400 # Number of points
            loss = loss.sum(dim=0) / cfg['test']['batch_size']
            
            running_loss += loss.item()
            
    avg_loss = running_loss / len(val_loader)
    print(f"Validation Loss for Epoch {epoch+1}: {avg_loss:.4f}")
    
    if cfg['train']['wandb']['use']:
        wandb.log({"val_loss": avg_loss, "epoch": epoch})
        
    return avg_loss


if __name__ == '__main__':
    with open('Config/train_test.yaml', 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    if cfg['train']['wandb']['use']:
        wandb.init(project=cfg['train']['wandb']['project'], entity=cfg['train']['wandb']['entity'], config=cfg)
    
    model = Fusion(backbone=cfg['dataset'].get('backbone', 'resnet')).to(device)
    
    # loss and optimizer
    criterion = unsupervisedLoss
    optimizer = torch.optim.Adam(model.parameters(), 
                                 lr=cfg['train']['learning_rate'], 
                                 weight_decay=cfg['train']['weight_decay']
                                 )
    epoch = cfg['train']['num_epochs']
    batch_size = cfg['train']['batch_size']

    if args.mode == 'train':
        # load dataset
        train_set = KittiDataset(root=cfg['dataset']['root_dir'], mode='train') 
        train_loader = DataLoader(dataset=train_set, batch_size=batch_size, 
                                  shuffle=True, collate_fn=train_set.collate_fn
                                  )
        
        # Validation set (using test mode for now as per original code structure, or split train set)
        # Assuming 'val_split' in config points to a validation set
        val_set = KittiDataset(root=cfg['dataset']['root_dir'], mode='test') # Using test mode for validation as placeholder
        val_loader = DataLoader(dataset=val_set, batch_size=cfg['test']['batch_size'], 
                                 shuffle=False, collate_fn=val_set.collate_fn
                                 )

        loss_values = []
        best_val_loss = float('inf')
        save_dir = cfg['train'].get('save_dir', './checkpoints')
        os.makedirs(save_dir, exist_ok=True)

        for i in range(epoch):
            loss = train_one_epoch(model, train_loader, optimizer, i, cfg)
            loss_values.append(loss)
            
            val_loss = validate_one_epoch(model, val_loader, i, cfg)
            
            # Save checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                saveCheckpoint(model, i, optimizer, val_loss, os.path.join(save_dir, 'best_model.pth'))
                print(f"Saved best model with val loss: {best_val_loss:.4f}")
            
            # Save latest
            saveCheckpoint(model, i, optimizer, val_loss, os.path.join(save_dir, 'latest_model.pth'))
        
        print('Finished Training')
        # plt.plot(np.array(loss_values), 'r')
        # plt.show()

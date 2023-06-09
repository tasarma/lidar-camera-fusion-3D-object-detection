from typing import List, Union
import argparse

import yaml
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from Utils.visualization import visualize_result
from Backbone.pointfusion import Fusion
from DataProcess.kitti_dataset import KittiDataset


parser = argparse.ArgumentParser(description="Arg parser")
parser.add_argument("--mode", type=str, default='train')


args = parser.parse_args()


def log_print(info: str, log_f: Union[str, None]=None) -> None:
    print(info)
    if log_f is not None:
        print(info, file=log_f)


def unsupervisedLoss(pred_offsets, pred_scores, targets):
    eps = 1e-16
    weight = 0.1
    L1 = nn.SmoothL1Loss(reduction='none')
    loss_offset = L1(pred_offsets, targets) # [B x pnts x 8 x 3]
    loss_offset = torch.mean(loss_offset, dim=(2, 3)) # [B x pnts]
    loss = ((loss_offset * pred_scores) - (weight * torch.log(pred_scores + eps)))
    # loss = loss.mean() # [1]
    return loss

def saveCheckpoint(model, epoch, optimizer, loss, path):
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
    model.train()
    log_print(f'===============TRAIN EPOCH {epoch+1}================')
    running_loss = 0.
    last_loss = 0.
    
    for itr, batch in enumerate(train_loader):
        img, points = batch['roi_img'], batch['roi_pc']
        targets, gt_corners = batch['corner_offsets'], batch['gt_corners']
        
        img = torch.from_numpy(img).float()#.cuda(non_blocking=True).float()
        points = torch.from_numpy(points).float()#.cuda(non_blocking=True).float()
        targets = torch.from_numpy(targets).float()#.cuda(non_blocking=True).float()
        gt_corners = torch.from_numpy(gt_corners).float()#.cuda(non_blocking=True).float()

        optimizer.zero_grad()

        pred_offset, scores = model(img, points)
        
		# Unsupervised loss
        loss = 0
        loss = unsupervisedLoss(pred_offset, scores, targets)
        loss = loss.sum(dim=1) / 400 # Number of points
        loss = loss.sum(dim=0) / cfg['train']['batch_size']

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        # Finding anchor point and predicted offset based on maximum score
        # max_inds = scores.max(dim=1)[1].cpu().numpy()
        # p_offset = np.zeros((4, 8, 3))
        # anchor_points = np.zeros((4, 3))
        # truth_boxes = np.zeros((4, 8, 3))
        # for i in range(0, 4):
        #     p_offset[i] = pred_offset[i][max_inds[i]].cpu().detach().numpy()
        #     anchor_points[i] = points[i][max_inds[i]].cpu().numpy()
        #     truth_boxes[i] = gt_corners[i].cpu().numpy()
        
        # visualize_result(p_offset, anchor_points, truth_boxes)
        # loss_epoch = running_loss / cfg['train']['batch_size']
        if itr % 10 == 0 and itr != 0:
            last_loss = running_loss / 10 # loss per batch
            print(f"Epoch [{epoch}/{cfg['train']['num_epochs']}], Step [{itr}] Loss: {last_loss:.4f}")
            # saveCheckpoint(model, epoch, optimizer, loss, f'models/pointfusion_{itr}.pth')
            running_loss = 0
    
    print(f"Loss for Epoch {epoch+1} is {last_loss} and running_loss is {running_loss}")
    return last_loss


if __name__ == '__main__':
    model = Fusion().to(device)

    with open('Config/train_test.yaml', 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    
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
        # test_loader = DataLoader(dataset=test_set, batch_size=batch_size, 
        #                          shuffle=True, collate_fn=data.collate_fn
        #                          )

        loss_values = []
        for i in range(epoch):
            loss = train_one_epoch(model, train_loader, optimizer, i, cfg)
            loss_values.append(loss)
        
        print('Finished Training')
        plt.plot(np.array(loss_values), 'r')
        plt.show()

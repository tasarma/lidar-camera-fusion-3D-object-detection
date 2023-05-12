from typing import List, Union
import argparse

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from Backbone.pointfusion import Fusion
from DataProcess.kitti_dataset import KittiDataset


parser = argparse.ArgumentParser(description="Arg parser")
parser.add_argument("--mode", type=str, default='train')


args = parser.parse_args()


def log_print(info: str, log_f: Union[str, None]=None) -> None:
    print(info)
    if log_f is not None:
        print(info, file=log_f)


def unsupervisedLoss(pred_offsets, pred_scores, offsets):
    eps = 1e-16
    weight = 0.1
    L1 = nn.SmoothL1Loss(reduction='none')
    loss_offset = L1(pred_offsets, offsets) # [B x pnts x 8 x 3]
    loss_offset = torch.mean(loss_offset, (2, 3)) # [B x pnts]
    # [B x pnts]
    loss = ((loss_offset * pred_scores) - (weight * torch.log(pred_scores + eps)))
    loss = loss.mean() # [1]
    return loss


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_one_epoch(model, train_loader, optimizer, epoch):
    model.train()
    log_print(f'===============TRAIN EPOCH {epoch}================')

    for itr, batch in enumerate(train_loader):
        # optimizer.zero_grad()
        img, pc, cls_labels = batch['roi_img'], batch['roi_pc'], batch['cls_labels']


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
    epoch = 1 

    if args.mode == 'train':
        # load dataset
        train_set = KittiDataset(root=cfg['dataset']['root_dir'], mode='train') 
        train_loader = DataLoader(dataset=train_set, batch_size=cfg['train']['batch_size'], 
                                  shuffle=True, collate_fn=train_set.collate_fn
                                  )
        # test_loader = DataLoader(dataset=test_set, batch_size=batch_size, 
        #                          shuffle=True, collate_fn=data.collate_fn
        #                          )

        train_one_epoch(model, train_loader, optimizer, epoch)

from typing import List, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from Backbone.pointfusion import Fusion
from DataProcess.kitti_dataset import KittiDataset

def log_print(info: str, log_f: Union[str, None]=None) -> None:
    print(info)
    if log_f is not None:
        print(info, file=log_f)


def splitTrainTest(train_set: KittiDataset, split: int) -> List[int]:
    total_size = train_set.__len__()
    train_size = int(split * total_size)
    test_size = total_size - train_size
    return [train_size, test_size]


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

# hyperparameters
learning_rate = 1e-3
weight_decay = 1e-5
batch_size = 10
num_epochs = 20




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_one_epoch(model, train_loader, optimizer, epoch):
    model.train()
    log_print(f'===============TRAIN EPOCH {epoch}================')

    for itr, data in enumerate(train_loader):
        optimizer.zero_grad()

        img = data['img'].to(device) 
        pts = data['pts'].to(device)
        target = data['target'].to(device)

        img = torch.from_numpy(img).cuda(non_blocking=True).float()
        pts = torch.from_numpy(pts).cuda(non_blocking=True).float()
        target = torch.from_numpy(target).cuda(non_blocking=True).float()


if __name__ == '__main__':
    model = Fusion().to(device)
    
    # loss and optimizer
    criterion = unsupervisedLoss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    epoch = 1 
    
    # load dataset
    data = KittiDataset(root=r'/home/tasarma/Playground/Tez/dataset/one_sample') 
    train_set, test_set = torch.utils.data.random_split(data, splitTrainTest(data, 1))
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, 
                              shuffle=True, collate_fn=data.collate_fn
                              )
    # test_loader = DataLoader(dataset=test_set, batch_size=batch_size, 
    #                          shuffle=True, collate_fn=data.collate_fn
    #                          )

    train_one_epoch(model, train_loader, optimizer, epoch)

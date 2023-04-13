import torch
import torch.nn as nn

from Backbone.pointfusion import Fusion
from DataProcess.kitti_dataset import KittiDataset


def splitTrainTest(train_set, split):
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

# load dataset
data = KittiDataset(root=r'/home/tasarma/Playground/Tez/dataset/kitti') 
train_set, test_set = torch.utils.data.random_split(data, splitTrainTest(data, 0.8))
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Fusion().to(device)


# loss and optimizer
criterion = unsupervisedLoss
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


for epoch in range(num_epochs):
    for batch_idx, (img, cloud, target) in enumerate(train_loader):
        print(img.shape, cloud.shape, target)
        break

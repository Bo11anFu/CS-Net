import torch
import torch.nn as nn
from csnet.losses.loss_basic import SoftIoULoss, BceLoss
from csnet.models.singleframe.ISNet.train_ISNet import Get_gradientmask_nopadding


class EdgeLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(EdgeLoss, self).__init__()
        self.bce = BceLoss(reduction=reduction)
        self.soft_iou = SoftIoULoss(smooth=1, reduction=reduction)

    def forward(self, edge_out, mask):
        edge = torch.cat([mask, mask, mask], dim=1).float()  # b, 3, m, n
        gradmask = Get_gradientmask_nopadding()
        edge_gt = gradmask(edge)

        edge_loss = 10 * self.bce(edge_out, edge_gt) + self.soft_iou(edge_out, edge_gt)

        return edge_loss

import math
from functools import partial

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

def focal_loss(pred, target):
    pred = pred.permute(0, 2, 3, 1)

    # -------------------------------------------------------------------------#
    #   找到每张图片的正样本和负样本
    #   一个真实框对应一个正样本
    #   除去正样本的特征点，其余为负样本
    # -------------------------------------------------------------------------#
    pos_inds = target.eq(1).float()
    neg_inds = target.lt(1).float()
    # -------------------------------------------------------------------------#
    #   正样本特征点附近的负样本的权值更小一些
    # -------------------------------------------------------------------------#
    neg_weights = torch.pow(1 - target, 4)

    pred = torch.clamp(pred, 1e-6, 1 - 1e-6)
    # -------------------------------------------------------------------------#
    #   计算focal loss。难分类样本权重大，易分类样本权重小。
    # -------------------------------------------------------------------------#
    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    # -------------------------------------------------------------------------#
    #   进行损失的归一化
    # -------------------------------------------------------------------------#
    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = -neg_loss
    else:
        loss = -(pos_loss + neg_loss) / num_pos
    return loss


def reg_l1_loss(pred, target, mask):
    # --------------------------------#
    #   计算l1_loss
    # --------------------------------#
    pred = pred.permute(0, 2, 3, 1)
    expand_mask = torch.unsqueeze(mask, -1).repeat(1, 1, 1, 2)

    loss = F.l1_loss(pred * expand_mask, target * expand_mask, reduction='sum')
    loss = loss / (mask.sum() + 1e-4)
    return loss


def seg_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()

def SoftIoULoss(pred, target):
    # Old One
    pred = torch.sigmoid(pred)
    smooth = 1

    # print("pred.shape: ", pred.shape)
    # print("target.shape: ", target.shape)

    intersection = pred * target
    loss = (intersection.sum(axis=(1, 2, 3)) + smooth) / (pred.sum(axis=(1, 2, 3)) + target.sum(axis=(1, 2, 3)) - intersection.sum(axis=(1, 2, 3)) + smooth)

    # loss = (intersection.sum(axis=(1, 2, 3)) + smooth) / \
    #        (pred.sum(axis=(1, 2, 3)) + target.sum(axis=(1, 2, 3))
    #         - intersection.sum(axis=(1, 2, 3)) + smooth)
    loss = 1 - loss.mean()
    # loss = (1 - loss).mean()

    return loss

# def dice_loss(predictive, target, ep=1e-8):
#     intersection = 2 * torch.sum(predictive * target) + ep
#     union = torch.sum(predictive) + torch.sum(target) + ep
#     loss = 1 - intersection / union
#     return loss

def dice_loss(pred,target):
    # pred = pred.sigmoid()
    smooth = 0.00

    intersection = pred * target

    intersection_sum = torch.sum(intersection, dim=(1, 2, 3))
    pred_sum = torch.sum(pred, dim=(1, 2, 3))
    target_sum = torch.sum(target, dim=(1, 2, 3))
    loss = (intersection_sum + smooth) / \
               (pred_sum + target_sum - intersection_sum + smooth)

    loss = 1 - torch.mean(loss)
    return loss
class MTWHLoss(nn.Module):
    def __init__(self):
        super(MTWHLoss, self).__init__()
    def forward(self, inputs, targets, threshord=0.4):
        # 计算目标像素和背景像素的占比
        gt = (targets > threshord).float()
        pt = (inputs > threshord).float()
        Mt = (targets > 0.5).float()
        loss_dice = dice_loss(inputs, Mt)
        pos_ratio = torch.mean(Mt)
        neg_ratio = 1 - pos_ratio

        squared_difference = np.array(torch.abs(pt-gt).cpu(), dtype=np.float32)
        loss_ou = np.sum(squared_difference) / gt.size(0)
        weight = neg_ratio / np.maximum(pos_ratio.cpu(), 1e-6)
        # 创建BCELoss
        bce_loss = nn.BCELoss(weight=weight)

        # 计算带权重调整的Loss
        loss1 = bce_loss(inputs, Mt)

        loss = loss1 + loss_dice*10 + 0.2*loss_ou
        return loss




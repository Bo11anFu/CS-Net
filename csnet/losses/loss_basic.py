import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftIoULoss(nn.Module):
    def __init__(self, smooth=1, reduction='mean'):
        super(SoftIoULoss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, pred, mask):
        pred = torch.sigmoid(pred)
        intersection = pred * mask
        intersection_sum = torch.sum(intersection, dim=(1, 2, 3))
        pred_sum = torch.sum(pred, dim=(1, 2, 3))
        mask_sum = torch.sum(mask, dim=(1, 2, 3))
        iou = (intersection_sum + self.smooth) / (pred_sum + mask_sum - intersection_sum + self.smooth)
        if self.reduction == 'mean':
            return 1 - iou.mean()
        elif self.reduction == 'sum':
            return 1 - iou.sum()
        else:
            raise NotImplementedError(f'reduction type {self.reduction} not implemented')




#Original Loss

class MultiSoftIoULoss(nn.Module):
    def __init__(self, smooth=1.0, reduction='mean',
                 prior_weights=(0.8, 0.15, 0.05)):
        super().__init__()

        self.smooth = smooth
        self.reduction = reduction

        # ===== 固定先验=====
        prior = torch.tensor(prior_weights, dtype=torch.float32)
        prior = prior / prior.sum()
        self.register_buffer('log_prior', torch.log(prior))

        # ===== 可学习偏移（epoch1 内会被训练）=====
        self.scale_logits = nn.Parameter(torch.zeros(len(prior)))

        # ===== 冻结相关 =====
        self.freeze = False
        self.register_buffer('fixed_weights', prior.clone())

    def current_weights(self):
        """
        对 Trainer 暴露：当前真正使用的权重
        """
        if self.freeze:
            return self.fixed_weights
        return torch.softmax(self.log_prior + self.scale_logits, dim=0)

    def freeze_with_weights(self, weights: torch.Tensor):
        """
        在 epoch1 结束后调用
        """
        with torch.no_grad():
            self.fixed_weights.copy_(weights)
            self.freeze = True
            self.scale_logits.requires_grad_(False)

    def soft_iou(self, pred, target):
        pred = torch.sigmoid(pred)

        intersection = pred * target
        intersection_sum = intersection.sum(dim=(1, 2, 3))
        pred_sum = pred.sum(dim=(1, 2, 3))
        target_sum = target.sum(dim=(1, 2, 3))

        iou = (intersection_sum + self.smooth) / \
              (pred_sum + target_sum - intersection_sum + self.smooth)

        if self.reduction == 'mean':
            return 1.0 - iou.mean()
        elif self.reduction == 'sum':
            return 1.0 - iou.sum()
        else:
            raise NotImplementedError

    def forward(self, pred_list, mask):
        # pred_list: [384, 192, 96]，利用差值函数来执行尺寸下采样
        mask_384 = mask
        mask_192 = F.interpolate(mask, size=(192, 192), mode='bilinear', align_corners=True)
        mask_96  = F.interpolate(mask, size=(96, 96),  mode='bilinear', align_corners=True)

        masks = [mask_384, mask_192, mask_96]
        weights = self.current_weights()

        loss = 0.0
        for p, g, w in zip(pred_list, masks, weights):
            loss = loss + w * self.soft_iou(p, g)

        return loss



#DiceLoss计算的是Dice系数，也称为F1分数
class DiceLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(DiceLoss, self).__init__()
        self.reduction = reduction
        self.eps = 1e-6

    def forward(self, pred, mask):
        pred = torch.sigmoid(pred)
        intersection = torch.sum(pred * mask, dim=(1, 2, 3))
        total_sum = torch.sum((pred + mask), dim=(1, 2, 3))
        dice = 2 * intersection / (total_sum + self.eps)
        if self.reduction == 'mean':
            return 1 - dice.mean()
        elif self.reduction == 'sum':
            return 1 - dice.sum()
        else:
            raise NotImplementedError(f'reduction type {self.reduction} not implemented')


class BceLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(BceLoss, self).__init__()
        self.reduction = reduction

    def forward(self, pred, mask):
        loss_fn = nn.BCEWithLogitsLoss(reduction=self.reduction)
        return loss_fn(pred, mask)


class L1Loss(nn.Module):
    def __init__(self, reduction='mean'):
        super(L1Loss, self).__init__()
        self.reduction = reduction

    def forward(self, pred, mask):
        loss_fn = nn.L1Loss(reduction=self.reduction)
        return loss_fn(pred, mask)

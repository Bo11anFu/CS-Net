import torch
import torch.nn as nn

#最大化I(v; z|y)
#特征 v (未经MFB)：知识渊博的**“教师”**，它看到了所有原始信息，能给出一个比较好的解答（pred_v）。
#特征 z (经过MFB)：被要求精简笔记的**“学生”**，它的知识是老师知识的压缩版。
class SufficiencyLoss(nn.Module):
    def __init__(self, temperature=1, reduction='mean'):
        super(SufficiencyLoss, self).__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.softmax = torch.nn.Softmax(dim=1)
        self.log_softmax = torch.nn.LogSoftmax(dim=1)
        self.weights = [1, 1, 1, 1, 1]

    def forward(self, pred_z_list, pred_v_list, mask):
        num_preds = len(pred_z_list)
        sufficiency_loss = 0
        for i in range(num_preds):
            pred_z = pred_z_list[i]#P_z
            pred_v = pred_v_list[i]#P_v

            b, c, _, _ = pred_z.shape

            #  对每个阶段的“能否正确预测前景/背景”进行监督，确保预测的前景分布与掩码一致
            ce = nn.BCEWithLogitsLoss(reduction=self.reduction)
            ce_loss = ce(pred_z, mask)

            pred_z = pred_z.reshape(b * c, -1)
            pred_v = pred_v.reshape(b * c, -1)

            #知识蒸馏实现DKL（P_v||P_z）
            #.detach(),作用为不要计算教师P_v的梯度让学生P_z去模仿教师P_v
            kld = torch.nn.KLDivLoss()
            kld_loss = kld(self.log_softmax(pred_v.detach() / self.temperature),
                           self.softmax(pred_z / self.temperature))

            loss = (kld_loss + ce_loss) / 2
            sufficiency_loss += (loss * self.weights[i])

        return sufficiency_loss / num_preds

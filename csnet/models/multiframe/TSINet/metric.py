import  numpy as np
import torch.nn as nn
import torch
from skimage import measure
import  numpy
class ROCMetric():
    """Computes pixAcc and mIoU metric scores
    """
    def __init__(self, nclass, bins):  #bin的意义实际上是确定ROC曲线上的threshold取多少个离散值
        super(ROCMetric, self).__init__()
        self.nclass = nclass
        self.bins = bins
        self.tp_arr = np.zeros(self.bins+1)
        self.pos_arr = np.zeros(self.bins+1)
        self.fp_arr = np.zeros(self.bins+1)
        self.neg_arr = np.zeros(self.bins+1)
        self.class_pos=np.zeros(self.bins+1)
        # self.reset()

    def update(self, preds, labels):
        for iBin in range(self.bins+1):
            score_thresh = (iBin + 0.0) / self.bins
            # print(iBin, "-th, score_thresh: ", score_thresh)
            i_tp, i_pos, i_fp, i_neg,i_class_pos = cal_tp_pos_fp_neg(preds, labels, self.nclass,score_thresh)
            self.tp_arr[iBin]   += i_tp
            self.pos_arr[iBin]  += i_pos
            self.fp_arr[iBin]   += i_fp
            self.neg_arr[iBin]  += i_neg
            self.class_pos[iBin]+=i_class_pos

    def get(self):

        tp_rates    = self.tp_arr / (self.pos_arr + 0.001)
        fp_rates    = self.fp_arr / (self.neg_arr + 0.001)

        recall      = self.tp_arr / (self.pos_arr   + 0.001)
        precision   = self.tp_arr / (self.class_pos + 0.001)


        return tp_rates, fp_rates, recall, precision

    def reset(self):

        self.tp_arr   = np.zeros([11])
        self.pos_arr  = np.zeros([11])
        self.fp_arr   = np.zeros([11])
        self.neg_arr  = np.zeros([11])
        self.class_pos= np.zeros([11])



class PD_FA():
    def __init__(self, nclass, bins):
        super(PD_FA, self).__init__()
        self.nclass = nclass
        self.bins = bins
        self.image_area_total = []
        self.image_area_match = []
        self.FA = np.zeros(self.bins+1)
        self.PD = np.zeros(self.bins + 1)
        self.target= np.zeros(self.bins + 1)
    def update(self, preds, labels):

        for iBin in range(self.bins+1):
            score_thresh = iBin * (255/self.bins)
            predits  = np.array((preds > score_thresh).cpu()).astype('int64')
            predits  = np.reshape (predits,  (256,256))
            labelss = np.array((labels).cpu()).astype('int64') # P
            labelss = np.reshape (labelss , (256,256))

            image = measure.label(predits, connectivity=2)
            coord_image = measure.regionprops(image)
            label = measure.label(labelss , connectivity=2)
            coord_label = measure.regionprops(label)

            self.target[iBin]    += len(coord_label)
            self.image_area_total = []
            self.image_area_match = []
            self.distance_match   = []
            self.dismatch         = []

            for K in range(len(coord_image)):
                area_image = np.array(coord_image[K].area)
                self.image_area_total.append(area_image)

            for i in range(len(coord_label)):
                centroid_label = np.array(list(coord_label[i].centroid))
                for m in range(len(coord_image)):
                    centroid_image = np.array(list(coord_image[m].centroid))
                    distance = np.linalg.norm(centroid_image - centroid_label)
                    area_image = np.array(coord_image[m].area)
                    if distance < 3:
                        self.distance_match.append(distance)
                        self.image_area_match.append(area_image)

                        del coord_image[m]
                        break

            self.dismatch = [x for x in self.image_area_total if x not in self.image_area_match]
            self.FA[iBin]+=np.sum(self.dismatch)
            self.PD[iBin]+=len(self.distance_match)

    def get(self,img_num):

        Final_FA =  self.FA / ((256 * 256) * img_num)
        Final_PD =  self.PD /self.target

        return Final_FA,Final_PD


    def reset(self):
        self.FA  = np.zeros([self.bins+1])
        self.PD  = np.zeros([self.bins+1])

class mIoU():

    def __init__(self, nclass):
        super(mIoU, self).__init__()
        self.nclass = nclass
        self.reset()

    def update(self, preds, labels):
        # print('come_ininin')

        correct, labeled = batch_pix_accuracy(preds, labels)
        inter, union = batch_intersection_union(preds, labels, self.nclass)
        self.total_correct += correct
        self.total_label += labeled
        self.total_inter += inter
        self.total_union += union


    def get(self):

        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()
        return pixAcc, mIoU

    def reset(self):

        self.total_inter = 0
        self.total_union = 0
        self.total_correct = 0
        self.total_label = 0




def cal_tp_pos_fp_neg(output, target, nclass, score_thresh):

    predict = (torch.sigmoid(output) > score_thresh).float()
    if len(target.shape) == 3:
        target = np.expand_dims(target.float(), axis=1)
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")

    intersection = predict * ((predict == target).float())

    tp = intersection.sum()
    fp = (predict * ((predict != target).float())).sum()
    tn = ((1 - predict) * ((predict == target).float())).sum()
    fn = (((predict != target).float()) * (1 - predict)).sum()
    pos = tp + fn
    neg = fp + tn
    class_pos= tp+fp

    return tp, pos, fp, neg, class_pos

def batch_pix_accuracy(output, target):

    if len(target.shape) == 3:
        target = np.expand_dims(target.float(), axis=1)
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")

    assert output.shape == target.shape, "Predict and Label Shape Don't Match"
    predict = (output > 0).float()
    pixel_labeled = (target > 0).float().sum()
    pixel_correct = (((predict == target).float())*((target > 0)).float()).sum()



    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled


def batch_intersection_union(output, target, nclass):

    mini = 1
    maxi = 1
    nbins = 1
    predict = (output > 0).float()
    if len(target.shape) == 3:
        target = np.expand_dims(target.float(), axis=1)
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")
    intersection = predict * ((predict == target).float())

    area_inter, _  = np.histogram(intersection.cpu(), bins=nbins, range=(mini, maxi))
    area_pred,  _  = np.histogram(predict.cpu(), bins=nbins, range=(mini, maxi))
    area_lab,   _  = np.histogram(target.cpu(), bins=nbins, range=(mini, maxi))
    area_union     = area_pred + area_lab - area_inter

    assert (area_inter <= area_union).all(), \
        "Error: Intersection area should be smaller than Union area"
    return area_inter, area_union

def IoU(pred, target):
    # Old One
    pred = torch.sigmoid(pred)

    intersection = pred * target

    smooth = 1

    iou = (intersection.sum(axis=(1, 2, 3)) + smooth) / (pred.sum(axis=(1, 2, 3)) + target.sum(axis=(1, 2, 3)) - intersection.sum(axis=(1, 2, 3)) +smooth)

    return iou.mean()

def calculateF1Measure(output_image, gt_image, thre, d=10):
    """
    计算True Positives (TP), True Negatives (TN), 和 False Positives (FP)

    参数：
    output_img: 预测的中心点概率的二值图像，形状为 (H, W)，H和W分别为图像的高和宽
    gt_img: 实际的中心位置的二值图像，形状为 (H, W)，H和W分别为图像的高和宽
    d: 用于判断中心点之间距离的阈值

    返回：
    TP: True Positives 的数量
    FN: False Negatives 的数量
    FP: False Positives 的数量
    """
    TP = 0
    FN = 0
    FP = 0

    output_image = np.squeeze(output_image)
    gt_image = np.squeeze(gt_image)
    output_img = output_image > thre
    gt_img = gt_image > thre
    # 对于每个实际的中心点，检查是否在d范围内有预测的中心点
    for i in range(gt_img.shape[0]):
        for j in range(gt_img.shape[1]):
            if gt_img[i, j] == 1:  # 实际为中心点
                has_predicted_center = False
                for m in range(max(0, i - d), min(output_img.shape[0], i + d + 1)):
                    for n in range(max(0, j - d), min(output_img.shape[1], j + d + 1)):
                        if output_img[m, n] == 1:  # 在d范围内找到了预测的中心点
                            has_predicted_center = True
                            break
                if has_predicted_center:
                    TP += 1
                else:
                    FN += 1
            else:  # 实际不是中心点
                if output_img[i, j] == 1:  # 错误地预测为中心点
                   FP += 1

    recall = TP/np.maximum(1, np.sum(TP + FN))
    prec = TP/np.maximum(1, np.sum(TP + FP))
    F1 = 2 * recall * prec / np.maximum(0.001, recall + prec)

    return prec, recall, F1

# def calculateF1Measure(output_image, gt_image, thre):
#     output_image = np.squeeze(output_image)
#     gt_image = np.squeeze(gt_image)
#     out_bin = output_image > thre
#     gt_bin = gt_image > thre
#     recall = np.sum(gt_bin*out_bin)/np.maximum(1, np.sum(gt_bin))
#     prec   = np.sum(gt_bin*out_bin)/np.maximum(1, np.sum(out_bin))
#     F1 = 2*recall*prec/np.maximum(0.001, recall+prec)
#     return prec, recall, F1


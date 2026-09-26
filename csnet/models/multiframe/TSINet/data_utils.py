from datetime import datetime

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data.dataset import Dataset
import random
import matplotlib.pyplot as plt
import os
import math
# import cv2

def make_dir(dataset, model):
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")

    save_dir = "%s_%s_%s_woDS" % (dataset, model, dt_string)
    os.makedirs('ablation/%s' % save_dir, exist_ok=True)
    return save_dir

def save_train_log(args, save_dir):
    dict_args=vars(args)
    args_key=list(dict_args.keys())
    args_value = list(dict_args.values())
    with open('ablation/%s/train_log.txt'%save_dir ,'w') as  f:
        now = datetime.now()
        f.write("time:--")
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        f.write(dt_string)
        f.write('\n')
        for i in range(len(args_key)):
            f.write(args_key[i])
            f.write(':--')
            f.write(str(args_value[i]))
            f.write('\n')
    return



class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_ckpt(state, save_path, filename):
    torch.save(state, os.path.join(save_path,filename))

def save_model(mean_F1, best_F1, save_dir, save_prefix, train_loss, recall, precision, epoch, net):
    if mean_F1 > best_F1:
        save_mF1_dir = 'result/' + save_dir + '/' + save_prefix + '_best_F1_F1.log'
        save_other_metric_dir = 'result/' + save_dir + '/' + save_prefix + '_best_F1_other_metric.log'
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        best_F1 = mean_F1
        save_model_and_result(dt_string, epoch, train_loss,  best_F1,
                              recall, precision, save_mIoU_dir, save_other_metric_dir)
        save_ckpt({
            'epoch': epoch,
            'state_dict': net,
            'mean_F1': mean_F1,
        }, save_path='result/' + save_dir,
            filename='mF1_' + '_' + save_prefix + '_epoch' + '.pth.tar')

def save_model_and_result(dt_string, epoch,train_loss, best_F1, recall, precision, save_mF1_dir, save_other_metric_dir):

    with open(save_mF1_dir, 'a') as f:
        f.write('{} - {:04d}:\t - train_loss: {:04f}:\t - mIoU {:.4f}\n' .format(dt_string, epoch,train_loss, best_F1))
    with open(save_other_metric_dir, 'a') as f:
        f.write(dt_string)
        f.write('-')
        f.write(str(epoch))
        f.write('\n')
        f.write('Recall-----:')
        for i in range(len(recall)):
            f.write('   ')
            f.write(str(round(recall[i], 8)))
            f.write('   ')
        f.write('\n')

        f.write('Precision--:')
        for i in range(len(precision)):
            f.write('   ')
            f.write(str(round(precision[i], 8)))
            f.write('   ')
        f.write('\n')
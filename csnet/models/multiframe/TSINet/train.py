import argparse
import os
import time
import torch
# import path
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from dataset import *
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from data_utils import *
from models.TSINet import Net
from metric import calculateF1Measure
from losses import *


parser = argparse.ArgumentParser(description="PyTorch MoCoPnet")
# parser.add_argument("--save", default='/omnisky3/pj/MoCoPnet-main/code_i/SIMTD', type=str, help="Save path")
parser.add_argument("--resume", default="", type=str, help="Resume path (default: none)")
parser.add_argument("--input_num", type=int, default=5, help="input frame number")
# parser.add_argument("--train_dataset_dir", default='/omnisky3/pj/demo/cloud_bg/imgs/train/', type=str, help="train_dataset")
parser.add_argument("--train_dataset_dir", default='',
                    type=str, help="train_dataset")
parser.add_argument("--test_dataset_dir", default='', type=str,
                    help="val_dataset")
parser.add_argument("--train_label_dir", default='', type=str,
                    help="train_label")
parser.add_argument("--test_label_dir", default='', type=str,
                    help="val_label")
parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
parser.add_argument('--n_iters', type=int, default=10, help='number of iterations to train')
parser.add_argument("--device", default=0, type=int, help="GPU id (default: 0)")
parser.add_argument("--lr", type=float, default=5e-3, help="Learning Rate. Default=4e-4")
parser.add_argument("--threads", type=int, default=0, help="Number of threads for data loader to use, Default: 1")

global opt
opt = parser.parse_args()
opt.save_dir = make_dir('SIATD', 'TSINet')
save_train_log(opt, opt.save_dir)

def train(train_loader):
    epoch = 20
    net = Net(opt.input_num)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if opt.resume:
        ckpt = torch.load(opt.resume, map_location='cuda:0')
        net.load_state_dict(ckpt['state_dict'])

    net.to(device)
    losses = AverageMeter()
    prec_best = 0.0
    count = 0
    optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr)
    criterion_BCE = MTWHLoss().to(device)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20*4675, eta_min=1e-6)
    for epo in range(0, epoch):
        prog_bar = tqdm(train_loader, desc="Epoch {}".format(epo + 1))
        for idx_iter, (input, label) in enumerate(prog_bar):
            net.train()
            input, label = Variable(input).to(device), Variable(label).to(device)
            label = label[:, opt.input_num - 1:opt.input_num, :, :].squeeze(2)
            pred = net(input)
            pred = pred.sigmoid()
            loss = criterion_BCE(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.update(loss.item(), pred.size(0))

            count += 1
            prog_bar.set_postfix({'train_loss': '{:.5f}'.format(losses.avg)})

            if count % 4675 == 0:
                F1, prec, recall = valid(net)
                print('iter:%d val_F1:%.5F val_prec:%.5f val_recall:%.5f' % (count, F1, prec, recall))
                if prec > prec_best:
                    prec_best = prec
                    fusion_model_file = os.path.join('F:\\code\\ablation', opt.save_dir,
                                                     'siatd1-epo%d_input%d_F1_%.4f_prec_%.4f_recall_%.4f.pth' % (epo + 1, opt.input_num, F1, prec, recall))
                    torch.save(net.state_dict(), fusion_model_file)


def demo_test(net, test_set):

    F1_list = []
    prec_list = []
    recall_list = []
    with torch.no_grad():
        for step in range(opt.input_num - 1, len(test_set)):
            input, label = test_set[step]
            input, label = Variable(input).to(opt.device), Variable(label).to(opt.device)
            input = input.unsqueeze(0)
            label = label.unsqueeze(0)
            label = label[:, opt.input_num - 1:opt.input_num, :, :].squeeze(2)
            label = label.detach().cpu().numpy()
            if np.sum(label) == 0:
                continue
            pred = net(input)

            pred = pred.squeeze()
            pred = pred.sigmoid()

            prec, recall, F1 = calculateF1Measure(pred.detach().cpu().numpy(), label, 0.4)

            F1_list.append(F1)
            prec_list.append(prec)
            recall_list.append(recall)

    return sum(F1_list) / len(F1_list), sum(prec_list) / len(prec_list), sum(recall_list) / len(recall_list)


def valid(net):
    net.eval()
    F1_list = []
    prec_list = []
    recall_list = []
    video_list = os.listdir(opt.test_dataset_dir)
    for i in range(0, len(video_list)):
        video_name = video_list[i]
        test_set = TestSetLoader(opt.test_dataset_dir + '/' + video_name, opt.test_label_dir,
                            video_name, input_num=opt.input_num)
        F1, prec, recall = demo_test(net, test_set)
        F1_list.append(F1)
        prec_list.append(prec)
        recall_list.append(recall)
    mean_F1 = sum(F1_list) / len(F1_list)
    mean_prec = sum(prec_list) / len(prec_list)
    mean_recall = sum(recall_list) / len(recall_list)
    return mean_F1, mean_prec, mean_recall


def save_checkpoint(state, save_path, filename='checkpoint.pth.tar'):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(state, os.path.join(save_path, filename))



def main():
    train_set = TrainSetLoader(opt)
    train_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batch_size, shuffle=True)
    train(train_loader)


if __name__ == '__main__':
    main()


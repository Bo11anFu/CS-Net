import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import yaml
from PIL import Image
from sklearn.metrics import auc
from tqdm import tqdm

from csnet.datasets import build_dataset, DataLoaderX, unpack_batch
from csnet.losses import build_loss
from csnet.metrics import mIoUMetric, nIoUMetric, PdFaMetric, ROCMetric
from csnet.models import build_model, run_model
from csnet.utils import (ordered_yaml, set_optimizer, set_lr_scheduler, update_lr, get_current_lr, reset_loss_dict,
                            get_loss_dict, set_logger, log_train_iter_info, log_train_info, log_test_info, make_dir)



def set_seed(seed, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if cuda_deterministic:
        # slower, more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # faster
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


class Trainer(object):
    def __init__(self, args):
        # seed
        set_seed(args.seed)

        # config
        with open(args.config, mode='r') as f:
            self.cfg = yaml.load(f, Loader=ordered_yaml()[0])

        # logger
        self.logger, self.tb_logger = set_logger(self.cfg)

        # device
        self.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

        # dataloader
        self.train_dataset, self.val_dataset, self.val_hard_dataset = build_dataset(self.cfg['dataset'], mode='train')
        self.train_loader = DataLoaderX(self.train_dataset, batch_size=self.cfg['train']['batch_size'], shuffle=True,
                                        num_workers=self.cfg['train']['num_workers'], pin_memory=True, drop_last=True)
        self.val_loader = DataLoaderX(self.val_dataset, batch_size=1,
                                      num_workers=self.cfg['train']['num_workers'], pin_memory=True)
        self.val_hard_loader = DataLoaderX(self.val_hard_dataset, batch_size=1,
                                           num_workers=self.cfg['train']['num_workers'], pin_memory=True)
        self.freeze_iou_weight_epoch = 5
        self.iou_weight_ema = None
        # Original = 0.9
        self.iou_weight_ema_momentum = 0.97  
        

        # model
        self.model, self.model_name = build_model(self.cfg['model']['network'])
        if args.DataParallel:
            self.model = nn.DataParallel(self.model)
        self.model = self.model.to(self.device)

       # loss
        self.loss_fn, self.loss_weight, self.train_loss, self.train_iter_loss, self.test_loss, self.use_sufficiency_loss, self.use_edge_loss = \
            build_loss(self.cfg['model']['loss'])
        for loss_type, criterion in self.loss_fn.items():
            self.loss_fn[loss_type] = criterion.to(self.device)

        # optimizer
        optim_params = list(self.model.parameters())
        

        # 检查损失函数中是否有可学习参数
        for loss_type, criterion in self.loss_fn.items():
            if hasattr(criterion, 'parameters'):
                optim_params.extend(list(criterion.parameters()))

        self.optimizer, self.init_lr = set_optimizer(optim_params, self.cfg['model']['optimizer'])

        # lr scheduler
        self.total_epochs = self.cfg['train']['total_epochs']
        self.iters_per_epoch = len(self.train_loader)
        self.total_iters = self.total_epochs * self.iters_per_epoch
        self.lr_scheduler, self.step_interval, self.warmup_iters = \
            set_lr_scheduler(self.optimizer, self.total_epochs, self.total_iters, self.cfg['model']['lr_scheduler'])

 

        # metric
        self.mIoU_metric = mIoUMetric()
        self.nIoU_metric = nIoUMetric()
        self.PdFa_metric = PdFaMetric()
        self.bins = 10
        self.ROC_metric = ROCMetric(bins=self.bins)

        # resume
        self.cur_iter = 0
        self.cur_epoch = 1
        if self.cfg['train'].get('resume'):
            checkpoint = torch.load(self.cfg['train']['resume'], map_location='cpu')
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            self.cur_epoch = checkpoint['epoch'] + 1
            self.cur_iter = checkpoint['epoch'] * self.iters_per_epoch
            self.logger.info(f"resume training from epoch: {checkpoint['epoch']}")

        # other settings
        self.log_interval = self.cfg['train']['log_interval']
        self.save_interval = self.cfg['train']['save_interval']
        self.val_interval = self.cfg['train']['val_interval']
        self.pred_vis_dir = os.path.join(self.cfg['train']['exp_root'], 'pred_vis')
        self.checkpoint_dir = os.path.join(self.cfg['train']['exp_root'], 'checkpoints')
        make_dir(self.pred_vis_dir)
        make_dir(self.checkpoint_dir)
        self.validation_history = []

    def train(self, epoch):
        self.logger.info(f"[{self.cfg['train']['exp_name']}][epoch:{epoch}][trainset]\n")
        self.model.train()
        epoch_st_time = iter_st_time = time.time()
        for iter_idx, data in enumerate(self.train_loader): #处理一个Batchsize为一个iter iter_idx从0开始为当前epoch处理了多少个batchsize
            self.cur_iter += 1      #全局迭代次数
            # update learning rate
            update_lr(self.optimizer, self.init_lr, self.lr_scheduler, self.step_interval, self.warmup_iters,
                      self.cur_iter, iter_idx)

            # optimize one iter
            frames, mask, _, _, _ = unpack_batch(data)
            # frames, mask, name = data
            frames, mask = frames.to(self.device), mask.to(self.device)
            if self.use_sufficiency_loss:
                preds, pred_z_list, pred_v_list = run_model(self.model, self.model_name, self.use_sufficiency_loss,
                                                            self.use_edge_loss, frames)
            elif self.use_edge_loss:
                preds, edge_out = run_model(self.model, self.model_name, self.use_sufficiency_loss,
                                            self.use_edge_loss, frames)
            else:
                preds = run_model(self.model, self.model_name, self.use_sufficiency_loss, self.use_edge_loss, frames)
            total_loss = 0.  # total loss of one iter
            if not isinstance(preds, (list, tuple)):
                preds = [preds]
            
            # print("pred shape:", preds[0].shape)
            # print("pred shape:", preds[1].shape)
            # print("pred shape:", preds[2].shape)
            # print("pred shape:", preds[3].shape)
            # print("mask shape:", mask.shape)
            
            #loss.detach().clone(), 把 tensor 从计算图分离并复制
            #遍历所有损失函数进行计算和组合
            for loss_type, criterion in self.loss_fn.items():
                if loss_type == 'SufficiencyLoss':
                    #计算SufficiencyLoss并乘以权重
                    loss = criterion(pred_z_list, pred_v_list, mask) * self.loss_weight[loss_type]                  
                    self.train_loss[loss_type] += loss.detach().clone()#更新
                    self.train_iter_loss[loss_type] += loss.detach().clone()
                    total_loss += loss#累加到总损失
                elif loss_type == 'EdgeLoss':
                    loss = criterion(edge_out, mask) * self.loss_weight[loss_type]
                    self.train_loss[loss_type] += loss.detach().clone()
                    self.train_iter_loss[loss_type] += loss.detach().clone()
                    total_loss += loss
                elif loss_type == 'MultiSoftIoULoss':
                    loss = criterion(preds, mask) * self.loss_weight[loss_type][0]  
                    self.train_loss[loss_type + '_0'] += loss.detach().clone()
                    self.train_iter_loss[loss_type + '_0'] += loss.detach().clone()
                    total_loss += loss
                else:
                    for i, pred in enumerate(preds):
                        if loss_type in ['SLSIoULoss', 'SDMLoss']:
                            loss = criterion(pred, mask, epoch) * self.loss_weight[loss_type][i]
                        else:
                            #计算其他损失函数并乘以权重
                            loss = criterion(pred, mask) * self.loss_weight[loss_type][i]
                        #带索引的损失更新，为每个尺度的输出单独记录损失
                        self.train_loss[loss_type + '_' + str(i)] += loss.detach().clone()
                        self.train_iter_loss[loss_type + '_' + str(i)] += loss.detach().clone()
                        total_loss += loss#累加到总损失
            self.optimizer.zero_grad()
            total_loss.backward()

            # ========== 分层梯度裁剪 ==========
            # 收集模型参数
            model_params = [p for p in self.model.parameters() if p.grad is not None]

            # 收集损失函数中的可学习参数
            loss_params = []
            for loss_type, criterion in self.loss_fn.items():
                if hasattr(criterion, 'parameters'):
                    for p in criterion.parameters():
                        if p.grad is not None:
                            loss_params.append(p)

            # 分别裁剪 模型主参数采用宽松方式裁剪，Loss权重参数采用严格方式裁剪
            if model_params:
                torch.nn.utils.clip_grad_norm_(model_params, max_norm=1.0)
            if loss_params:
                torch.nn.utils.clip_grad_norm_(loss_params, max_norm=0.5)

            self.optimizer.step()


            if 'MultiSoftIoULoss' in self.loss_fn:
                criterion = self.loss_fn['MultiSoftIoULoss']

                if (epoch == self.freeze_iou_weight_epoch) and (not criterion.freeze):
                    with torch.no_grad():
                        w = criterion.current_weights()

                        if self.iou_weight_ema is None:
                            self.iou_weight_ema = w.clone()
                        else:
                            m = self.iou_weight_ema_momentum
                            self.iou_weight_ema = m * self.iou_weight_ema + (1 - m) * w
            # log iters within an interval
            if (iter_idx + 1) % self.cfg['train']['log_interval'] == 0:
                iter_time = time.time() - iter_st_time
                log_train_iter_info(epoch, iter_idx + 1, self.iters_per_epoch, get_current_lr(self.optimizer),
                                    self.cfg['train']['log_interval'], iter_time, get_loss_dict(self.train_iter_loss),
                                    self.logger)
                iter_st_time = time.time()

                if 'MultiSoftIoULoss' in self.loss_fn:
                    criterion = self.loss_fn['MultiSoftIoULoss']

                    with torch.no_grad():
                        if criterion.freeze:
                            w = criterion.current_weights()
                        else:
                            w = criterion.current_weights()

                    self.tb_logger.add_scalars(
                        'loss_weights',
                        {
                            '384': w[0].item(),
                            '192': w[1].item(),
                            '96':  w[2].item(),
                        },
                        self.cur_iter
                    )

                    self.logger.info(
                        f"MultiSoftIoULoss weights - "
                        f"384: {w[0].item():.6f}, "
                        f"192: {w[1].item():.6f}, "
                        f"96: {w[2].item():.6f}"
                    )

                        
        if 'MultiSoftIoULoss' in self.loss_fn:
                criterion = self.loss_fn['MultiSoftIoULoss']
                if (epoch == self.freeze_iou_weight_epoch) and (not criterion.freeze):
                    with torch.no_grad():
                        criterion.fixed_weights.copy_(self.iou_weight_ema)
                        criterion.freeze = True
                        criterion.scale_logits.requires_grad_(False)

                    self.logger.info(
                        f"[Freeze EMA] MultiSoftIoULoss weights fixed at epoch {epoch}: "
                        f"{self.iou_weight_ema.tolist()}"
                    )

        reset_loss_dict(self.train_iter_loss)

        # log one epoch
        epoch_time = time.time() - epoch_st_time
        log_train_info(epoch, get_current_lr(self.optimizer), epoch_time, get_loss_dict(self.train_loss),
                       self.logger, self.tb_logger)
        

    def validate(self, epoch, split='all'):
        self.logger.info(f"[{self.cfg['train']['exp_name']}][epoch:{epoch}][testset]\n")
        self.model.eval()
        # reset metrics
        self.mIoU_metric.reset()
        self.nIoU_metric.reset()
        self.PdFa_metric.reset()
        self.ROC_metric.reset()
        # start_time = time.time()

        if split == 'all':
            val_loader = self.val_loader
        elif split == 'hard':
            val_loader = self.val_hard_loader
        else:
            raise ValueError(f"Invalid split '{split}'. It must be 'all' or 'hard'.")

        with torch.no_grad():
            for iter_idx, data in enumerate(tqdm(val_loader)):
                frames, mask, h, w, name = unpack_batch(data)
                # frames, mask, name = data
                frames, mask = frames.to(self.device), mask.to(self.device)
                if self.use_sufficiency_loss:
                    preds, pred_z_list, pred_v_list = run_model(self.model, self.model_name, self.use_sufficiency_loss,
                                                                self.use_edge_loss, frames)
                elif self.use_edge_loss:
                    preds, edge_out = run_model(self.model, self.model_name, self.use_sufficiency_loss,
                                                self.use_edge_loss, frames)
                else:
                    preds = run_model(self.model, self.model_name, self.use_sufficiency_loss,
                                      self.use_edge_loss, frames)
                if not isinstance(preds, (list, tuple)):
                    preds = [preds]
                if split == 'all':
                    for loss_type, criterion in self.loss_fn.items():
                        if loss_type == 'SufficiencyLoss':
                            loss = criterion(pred_z_list, pred_v_list, mask) * self.loss_weight[loss_type]
                            self.test_loss[loss_type] += loss.detach().clone()
                        elif loss_type == 'EdgeLoss':
                            loss = criterion(edge_out, mask) * self.loss_weight[loss_type]
                            self.test_loss[loss_type] += loss.detach().clone()
                        elif loss_type == 'MultiSoftIoULoss':
                            loss = criterion(preds, mask) * self.loss_weight[loss_type][0]
                            self.test_loss[loss_type + '_0'] += loss.detach().clone()
                        else:
                            for i, pred in enumerate(preds):
                                if loss_type in ['SLSIoULoss', 'SDMLoss']:
                                    loss = criterion(pred, mask, epoch) * self.loss_weight[loss_type][i]
                                else:
                                    loss = criterion(pred, mask) * self.loss_weight[loss_type][i]
                                self.test_loss[loss_type + '_' + str(i)] += loss.detach().clone()

                pred = preds[0]  # Note: distinguish between 0 and -1 when using deep supervision
                # Restore the original image size
                pred = pred[:, :, :h, :w]
                mask = mask[:, :, :h, :w]

                # update metrics
                self.mIoU_metric.update(pred, mask)
                self.nIoU_metric.update(pred, mask)
                self.PdFa_metric.update(pred, mask)
                self.ROC_metric.update(pred, mask)

                # visualize predicted masks (only the last epoch)
                if split == 'all' and epoch == self.total_epochs:
                    pred_sigmoid = torch.sigmoid(pred)
                    final_pred = pred_sigmoid.data.cpu().numpy()[0, 0, :, :]
                    mask_pred = Image.fromarray(np.uint8(final_pred * 255))
                    save_dir = os.path.join(self.pred_vis_dir, name[0].split('/')[0])
                    make_dir(save_dir)
                    mask_pred.save(os.path.join(self.pred_vis_dir, name[0]))

        # get metrics
        # print('FPS=%.3f' % ((iter_idx + 1) / (time.time() - start_time)))
        _, mIoU = self.mIoU_metric.get()
        nIoU, _ = self.nIoU_metric.get()
        FA, PD = self.PdFa_metric.get()
        tp_rates, fp_rates, recall, precision, f_score = self.ROC_metric.get()
        AUC = auc(fp_rates, tp_rates)
        precision_05 = precision[(self.bins + 1) // 2]
        recall_05 = recall[(self.bins + 1) // 2]
        f_score_05 = f_score[(self.bins + 1) // 2]


        if split == 'all':
            self.validation_history.append({
                'epoch': epoch,
                'mIoU': mIoU,
                'nIoU': nIoU,
                'PD': PD,
                'FA': FA,
                'AUC': AUC,
                'precision_05': precision_05,
                'recall_05': recall_05,
                'f_score_05': f_score_05,
            })




        # log the test info
        test_loss = get_loss_dict(self.test_loss) if split == 'all' else None
        log_test_info(epoch, test_loss, mIoU, nIoU, PD, FA, AUC, precision_05, recall_05,
                      f_score_05, self.logger, self.tb_logger, split)

    def save_model(self, epoch):
        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'epoch': epoch,
            'config': self.cfg
        }
        checkpoint_path = os.path.join(self.checkpoint_dir, f'model_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f'Successfully saved the checkpoint in {checkpoint_path}.\n')
    def log_final_validation_summary(self):
        self.logger.info("\n" + "="*80)
        self.logger.info(f"       FINAL VALIDATION RESULTS SUMMARY (Epochs >= 30)")
        self.logger.info("="*80)


        header = f"{'Epoch':<7} | {'mIoU (%)':<10} | {'nIoU (%)':<10} | {'PD (%)':<10} | {'FA (%)':<12} | {'AUC (%)':<10} | {'F-Score (%)':<10}"
        self.logger.info(header)
        self.logger.info("-" * len(header))

        for results in self.validation_history:
            log_line = (
                f"{results['epoch']:<7} | "#占七个字符宽度
                f"{results['mIoU'] * 100:9.3f} | "  # 转换为百分比并保留3位小数
                f"{results['nIoU'] * 100:9.3f} | "
                f"{results['PD'] * 100:9.3f} | "
                f"{results['FA'] * 100:11.6f} | " # FA费率可能很小,多保留几位小数
                f"{results['AUC'] * 100:9.3f} | "
                f"{results['f_score_05'] * 100:9.3f}"
            )
            self.logger.info(log_line)
        
        self.logger.info("="*80 + "\n")

def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    trainer = Trainer(args)
    for epoch in range(trainer.cur_epoch, trainer.total_epochs + 1):
        # train
        trainer.train(epoch)
        # save
        if epoch % trainer.save_interval == 0:
            trainer.save_model(epoch)
        # val
        if epoch >= 30 and epoch % trainer.val_interval == 0:  # for MIST
            # if epoch >= 10 and epoch % trainer.val_interval == 0:  # for NUDT-MIRSDT
            trainer.validate(epoch, split='all')
            # trainer.validate(epoch, split='hard')

    trainer.log_final_validation_summary()

def args_parser():
    parser = argparse.ArgumentParser(description='PyTorch CSNet Training')
    parser.add_argument('--config', type=str,
                        default='/home/ubuntu/data/zhengqinxu/object_detection/CSNet/configs/multiframe/CSNet/train_CSNet_NUDTMIRSDT.yaml',
                        help='path to config file')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--device', type=str, default='cuda', help='device (use cuda or cpu)')
    parser.add_argument('--DataParallel', default=True, help='use DataParallel or not')
    parser.add_argument('--gpu_ids', type=str, default='0', help='the ids of gpus')

    return parser.parse_args()


if __name__ == '__main__':
    args = args_parser()
    main(args)

import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import yaml
from PIL import Image
from sklearn.metrics import auc
from tqdm import tqdm
from scipy.io import savemat

from csnet.datasets import build_dataset, DataLoaderX, unpack_batch
from csnet.metrics import mIoUMetric, nIoUMetric, PdFaMetric, PdFaMetric1, ROCMetric
from csnet.models import build_model, run_model
from csnet.utils import ordered_yaml, make_dir


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

        # device
        self.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

        # dataloader
        self.val_dataset, self.val_lSCR_dataset, self.val_hSCR_dataset = build_dataset(self.cfg['dataset'], mode='val')
        self.val_loader = DataLoaderX(self.val_dataset, batch_size=1,
                                      num_workers=self.cfg['test']['num_workers'], pin_memory=True)
        self.val_lSCR_loader = DataLoaderX(self.val_lSCR_dataset, batch_size=1,
                                           num_workers=self.cfg['test']['num_workers'], pin_memory=True)
        self.val_hSCR_loader = DataLoaderX(self.val_hSCR_dataset, batch_size=1,
                                           num_workers=self.cfg['test']['num_workers'], pin_memory=True)

        # model
        self.model, self.model_name = build_model(self.cfg['model']['network'])
        if args.DataParallel:
            self.model = nn.DataParallel(self.model)
        self.model = self.model.to(self.device)

        # metric
        self.mIoU_metric = mIoUMetric()
        self.nIoU_metric = nIoUMetric()
        self.PdFa_metric = PdFaMetric()
        self.bins = 10  # 10, 200
        self.ROC_metric = ROCMetric(bins=self.bins)

        # checkpoint
        checkpoint = torch.load(self.cfg['test']['checkpoint'], map_location='cpu')
        self.model.load_state_dict(checkpoint['model'])
        print(f"Successfully load the checkpoint from {self.cfg['test']['checkpoint']}.\n")

        # visualize
        self.pred_result_visualize_dir = os.path.join(self.cfg['test']['exp_root'], 'pred_result_visualize')
        make_dir(self.pred_result_visualize_dir)

    def testing(self, partition='all'):
        print(f"[testset]\n")
        self.model.eval()
        # reset metrics
        self.mIoU_metric.reset()
        self.nIoU_metric.reset()
        self.PdFa_metric.reset()
        self.ROC_metric.reset()

        if partition == 'all':
            val_loader = self.val_loader
        elif partition == 'lSCR':
            val_loader = self.val_lSCR_loader
        elif partition == 'hSCR':
            val_loader = self.val_hSCR_loader
        else:
            raise ValueError(f"Invalid partition '{partition}'. It must be 'all', 'lSCR' or 'hSCR'.")

        with torch.no_grad():
            for iter_idx, data in enumerate(tqdm(val_loader)):
                frames, mask, h, w, name = unpack_batch(data)
                frames, mask = frames.to(self.device), mask.to(self.device)
                if hasattr(self.model.module, 'use_ib_loss') and self.model.module.use_ib_loss:
                    preds, _, _ = run_model(self.model, self.model_name, True, frames)
                else:
                    preds = run_model(self.model, self.model_name, False, frames)

                if not isinstance(preds, (list, tuple)):
                    preds = [preds]
                pred = preds[0]  # Note: distinguish between 0 and -1 when using deep supervision
                # Restore the original image size
                pred = pred[:, :, :h, :w]
                mask = mask[:, :, :h, :w]

                # update metrics
                self.mIoU_metric.update(pred, mask)
                self.nIoU_metric.update(pred, mask)
                self.PdFa_metric.update(pred, mask)
                self.ROC_metric.update(pred, mask)

                # prediction result visualize
                if partition == 'all':
                    pred_sigmoid = torch.sigmoid(pred)
                    final_pred = pred_sigmoid.data.cpu().numpy()[0, 0, :, :]
                    mask_pred = Image.fromarray(np.uint8(final_pred * 255))
                    save_dir = os.path.join(self.pred_result_visualize_dir, name[0].split('/')[0])
                    make_dir(save_dir)
                    mask_pred.save(os.path.join(self.pred_result_visualize_dir, name[0]))

        # get metrics
        _, mIoU = self.mIoU_metric.get()
        nIoU, _ = self.nIoU_metric.get()
        FA, PD = self.PdFa_metric.get()
        tp_rates, fp_rates, recall, precision, f_score = self.ROC_metric.get()
        AUC = auc(fp_rates, tp_rates)
        precision_05 = precision[(self.bins + 1) // 2 + 1]
        recall_05 = recall[(self.bins + 1) // 2 + 1]
        f_score_05 = f_score[(self.bins + 1) // 2 + 1]

        # log the test info
        message = f'\n[{partition}]'
        message += f'[mIoU:{mIoU:.5f}][nIoU:{nIoU:.5f}]'
        message += f'[PD:{PD:.5f}][FA:{FA:.5e}]'
        message += f'[AUC:{AUC:.5f}][precision:{precision_05:.5f}][recall:{recall_05:.5f}][f_score:{f_score_05:.5f}]'
        print(message)

        # save mat
        save_dict = {'mIoU': mIoU, 'nIoU': nIoU,
                     'PD': PD, 'FA': FA,
                     'tp_rates': tp_rates, 'fp_rates': fp_rates, 'AUC': AUC,
                     'precision': precision_05, 'recall': recall_05, 'f_score': f_score_05}
        savemat(os.path.join(self.cfg['test']['exp_root'], f'{partition}_metrics.mat'), save_dict)


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    trainer = Trainer(args)
    trainer.testing(partition='all')
    trainer.testing(partition='lSCR')
    trainer.testing(partition='hSCR')


def args_parser():
    parser = argparse.ArgumentParser(description='PyTorch Deep-MIRST Test')
    parser.add_argument('--config', type=str,
                        default='./configs/test/TCPD/TCPD_IBLoss_NUDTMIRSDT.yaml',
                        help='path to config file')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--device', type=str, default='cuda', help='device (use cuda or cpu)')
    parser.add_argument('--DataParallel', default=True, help='use DataParallel or not')
    parser.add_argument('--gpu_ids', type=str, default='0', help='the ids of gpus')

    return parser.parse_args()


if __name__ == '__main__':
    args = args_parser()
    main(args)

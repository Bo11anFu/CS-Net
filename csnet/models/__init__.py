import math
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from copy import deepcopy
from csnet.models.singleframe.ACM.model_ACM import ASKCResUNet as ACM
from csnet.models.singleframe.ALCNet.model_ALCNet import ASKCResNetFPN as ALCNet
from csnet.models.singleframe.DNANet.model_DNANet import DNANet
from csnet.models.singleframe.FC3Net.model_FC3Net import FC3 as FC3Net
from csnet.models.singleframe.UIUNet.model_UIUNet import UIUNet
from csnet.models.singleframe.RDIAN.model_RDIAN import RDIAN
from csnet.models.singleframe.MiM.model_MiM import MiM
from csnet.models.singleframe.MSHNet.model_MSHNet import MSHNet
from csnet.models.multiframe.DTUM.model_ResUNet_DTUM import ResUNet_DTUM
from csnet.models.multiframe.DTUM.model_ALCNet_DTUM import ALCNet_DTUM
from csnet.models.multiframe.DTUM.model_DNANet_DTUM import DNANet_DTUM
from csnet.models.multiframe.DTUM.model_UIUNet_DTUM import UIUNet_DTUM
from csnet.models.multiframe.PSTFNet.model_PSTFNet import PSTFNet
#from csnet.models.multiframe.RFR.model_RFR import RFR  
from csnet.models.multiframe.LVNet.model_LVNet import LVNet

# from csnet.models.multiframe.MISTNet.model_MISTNet import MISTNet

from csnet.models.multiframe.DeepPro.model_DeepPro import DeepPro
from csnet.models.multiframe.RFR_STDNet.model_RFR_STDNet import RFR_STDNet
from csnet.models.multiframe.STDQNet.model_STDQNet import STDQNet
from csnet.models.multiframe.STDBNet.model_STDBNet import STDBNet
from csnet.models.multiframe.CSNet.model_CSNet import CSNet
from csnet.models.singleframe.NSFPN.model_NSFPN import MSHNet_NSFPN
from csnet.models.singleframe.DHiF.model_DNANet_DHiF import DNANet_DHiF
from csnet.models.singleframe.DHiF.model_MSHNet_DHiF import MSHNet_DHiF
from csnet.models.multiframe.ADSUNet.models.SiamUnet_conc_diff_cbam import ADSUNet
from csnet.models.multiframe.DQAligner.model.DQAligner import DQAligner
from csnet.models.multiframe.TSINet.models.TSINet import TSINet
from csnet.models.multiframe.DCPNet.model_DCPNet import DCPNet
from csnet.models.multiframe.DTUM.model_ResUNet_DTUM_vis_feat import ResUNet_DTUM_vis_feat



def init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
    elif isinstance(m, nn.Conv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        fan_out //= m.groups
        m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))

        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.Conv3d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
        fan_out //= m.groups
        m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


def build_model(model_cfg):
    model_cfg = deepcopy(model_cfg)
    model_name = model_cfg.pop('name')
    # single-frame
    if model_name == 'ACM':
        model = ACM(**model_cfg)
    elif model_name == 'ALCNet':
        model = ALCNet(**model_cfg)
    elif model_name == 'DNANet':
        model = DNANet(**model_cfg)
    elif model_name == 'FC3Net':
        model = FC3Net(**model_cfg)
    elif model_name == 'ISNet':
        model = ISNet(**model_cfg)
    elif model_name == 'UIUNet':
        model = UIUNet(**model_cfg)
    elif model_name == 'RDIAN':
        model = RDIAN(**model_cfg)
    elif model_name == 'MiM':
        model = MiM(**model_cfg)
    elif model_name == 'MSHNet':
        model = MSHNet(**model_cfg)
    # multi-frame
    elif model_name == 'ResUNet_DTUM':
        model = ResUNet_DTUM(**model_cfg)
    elif model_name == 'ALCNet_DTUM':
        model = ALCNet_DTUM(**model_cfg)
    elif model_name == 'DNANet_DTUM':
        model = DNANet_DTUM(**model_cfg)
    elif model_name == 'UIUNet_DTUM':
        model = UIUNet_DTUM(**model_cfg)
    elif model_name == 'PSTFNet':
        model = PSTFNet(**model_cfg)
    elif model_name == 'RFR':
        model = RFR(**model_cfg)
    elif model_name == 'LVNet':
        model = LVNet(**model_cfg)
    elif model_name == 'CSNet':
        model = CSNet(**model_cfg)
    elif model_name == 'DeepPro':
        model = DeepPro(**model_cfg)
    elif model_name == 'NSFPN':
        model = MSHNet_NSFPN(**model_cfg)
    elif model_name == 'DNANet_DHiF':
        model = DNANet_DHiF(**model_cfg)
    elif model_name == 'MSHNet_DHiF':
        model = MSHNet_DHiF(**model_cfg)
    elif model_name == 'ADSUNet':
        model = ADSUNet(**model_cfg)
    elif model_name == 'DQAligner':
        model = DQAligner(**model_cfg)
    elif model_name == 'TSINet':
        model = TSINet(**model_cfg)
    elif model_name == 'DCPNet':
        model = DCPNet(**model_cfg)
    elif model_name == 'ResUNet_DTUM_vis_feat':
        model = ResUNet_DTUM_vis_feat(**model_cfg)
    elif model_name == 'RFR_vis_feat':
        model = RFR_vis_feat(**model_cfg)
    else:
        raise NotImplementedError(f"Invalid model name '{model_name}'.")
    return model, model_name


def run_model(model, model_name, use_sufficiency_loss, use_edge_loss, frames):
    # single-frame
    if model_name in ['ACM', 'ALCNet', 'DNANet', 'FC3Net', 'UIUNet', 'RDIAN', 'MiM', 'MSHNet']:
        frames = frames[:, :, -1, :, :]
        preds = model(frames)
    elif model_name in ['ISNet']:
        frames = frames[:, :, -1, :, :]
        if use_edge_loss:
            preds, edge_out = model(frames)
            return preds, edge_out
        else:
            preds = model(frames)
    # multi-frame
    elif model_name in ['ResUNet_DTUM', 'ALCNet_DTUM', 'DNANet_DTUM', 'UIUNet_DTUM']:
        preds = model(frames)
        preds = torch.squeeze(preds, 2)
    elif model_name in ['PSTFNet', 'LVNet', 'MISTNet_wo_MFB_Ls']:
        preds = model(frames)
    elif model_name in ['RFR']:
        frames = frames.permute(0, 2, 1, 3, 4).contiguous()
        preds = model(frames)
    elif model_name in ['CSNet']:
        if use_sufficiency_loss:
            preds, pred_z_list, pred_v_list = model(frames)
            return preds, pred_z_list, pred_v_list
        else:
            preds = model(frames)
    elif model_name in [ 'ResUNet_DTUM_vis_feat']:
        preds, z_4, z_3, z_2, z_1 = model(frames)
        return preds, z_4, z_3, z_2, z_1
    elif model_name in ['RFR_vis_feat']:
        frames = frames.permute(0, 2, 1, 3, 4).contiguous()
        preds, z_4, z_3, z_2, z_1 = model(frames)
        return preds, z_4, z_3, z_2, z_1
    elif model_name in ['DeepPro']:
        seq_feats, preds = model(frames)
        preds = preds[:, -1, :, :].unsqueeze(1)
    else:
        raise NotImplementedError(f"Invalid model name '{model_name}'.")
    return preds

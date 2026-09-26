import torch
import math

import itertools
import torch.nn.functional as F

from torch.nn import init
from torch.nn.init import trunc_normal_
from torch.nn.modules.utils import _triple
import torch.nn as nn

from csnet.models.multiframe.STDBNet import base_STD as base

from thop import profile, clever_format
import functools
from csnet.models.multiframe.CSNet.gci_conv import GCIConv
from csnet.models.multiframe.CSNet.gci_conv_group import GCIConvGroupAgg


class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation, groups=self.c.groups,
            device=c.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m

class DSPConv(nn.Module):
    def __init__(self, dim, lks, sks, groups):
        super().__init__()
        self.cv1 = Conv2d_BN(dim, dim // 2)
        self.act = nn.ReLU()
        self.cv2 = Conv2d_BN(dim // 2, dim // 2, ks=lks, pad=(lks - 1) // 2, groups=dim // 2)
        self.cv3 = Conv2d_BN(dim // 2, dim // 2)
        self.cv4 = nn.Conv2d(dim // 2, sks ** 2 * dim // groups, kernel_size=1)
        self.norm = nn.GroupNorm(num_groups=dim // groups, num_channels=sks ** 2 * dim // groups)
        
        self.sks = sks
        self.groups = groups
        self.dim = dim
        
    def forward(self, x):
        x = self.act(self.cv3(self.cv2(self.act(self.cv1(x)))))
        #x = self.act(self.cv3(self.act(self.cv1(x))))
        w = self.norm(self.cv4(x))
        b, _, h, width = w.size()
        w = w.view(b, self.dim // self.groups, self.sks ** 2, h, width)

        return w

class DSPGConv(nn.Module):
    def __init__(self, dim, G, lks=5, sks=3):
        super(DSPGConv, self).__init__()
        self.dim = dim
        self.G = G
        self.sks = sks

        mid = dim // 2

        self.cv1 = Conv2d_BN(dim, mid)
        self.act = nn.ReLU(inplace=True)
        self.cv2 = Conv2d_BN(mid, mid, ks=lks, pad=(lks-1)//2, groups=mid)
        self.cv3 = Conv2d_BN(mid, mid)

        # output channels now = G * (ks*ks)
        self.cv4 = nn.Conv2d(mid, G * (sks*sks), kernel_size=1)
        self.norm = nn.GroupNorm(num_groups=G, num_channels=G * (sks*sks))

    def forward(self, x):
        x = self.act(self.cv1(x))
        x = self.act(self.cv2(x))
        x = self.act(self.cv3(x))

        w = self.cv4(x)
        B, _, H, W = w.shape
        w = self.norm(w)
        return w.view(B, self.G, self.sks*self.sks, H, W)


class CSHD(nn.Module):
    def __init__(self, dim):
        super(CSHD, self).__init__()
        self.dsp_conv = DSPConv(dim, lks=7, sks=3, groups=8)
        self.gci_conv = GCIConv()
        self.bn = nn.BatchNorm2d(dim)

    def forward(self, x):
        
        return self.bn(self.gci_conv(x, self.dsp_conv(x))) + x

class CSHDG(nn.Module):
    def __init__(self, in_channel, out_channel, G=8, lks=7, sks=3):
        super(CSHDG, self).__init__()

        assert in_channel % G == 0
        assert out_channel % G == 0

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.G = G

        self.dsp_conv = DSPGConv(in_channel, G=G, lks=lks, sks=sks)

        self.gci_conv = GCIConvGroupAgg()
        self.bn = nn.BatchNorm2d(out_channel)

        if in_channel != out_channel:
            self.res = nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False)
            self.res_bn = nn.BatchNorm2d(out_channel)
        else:
            self.res = None

    def forward(self, x):
        w = self.dsp_conv(x)  # (B, G, ks*ks, H, W)
        o = self.gci_conv(x, w, self.out_channel)
        o = self.bn(o)
        if self.res is not None:
            return o + self.res_bn(self.res(x))
        else:
            return o + x



class CSHDBlock(nn.Module):
    def __init__(self, num_feat):
        super(CSHDBlock, self).__init__()

        in_ch = 2 * num_feat

        #采用CSHD KL=7 KS=3 Group=8
        self.cshd = CSHD(in_ch)
        self.PWconv = nn.Conv2d(in_ch, num_feat, kernel_size=1,bias=False)
        self.act = nn.ReLU(inplace=False)



    def forward(self, past_feat, curr_feat):
        x = torch.cat([past_feat, curr_feat], dim=1)

        x=self.cshd(x)
        x=self.PWconv(x)

        return x




class MSAM(nn.Module):
    def __init__(self, block, nb_filter, num_frames, num_levels):
        super(MSAM, self).__init__()
        self.num_levels = num_levels
        self.aggregators = nn.ModuleList([

            base.make_layer(block, nb_filter[j] * num_frames, nb_filter[j])
            for j in range(num_levels)
        ])

    def forward(self, compensated_feats):
        return [self.aggregators[j](compensated_feats[j]) for j in range(self.num_levels)]


#采用Unet解码器 BaseDecoder
class BaseDecoder(nn.Module):
    def __init__(self, num_classes, block, nb_filter):
        super(BaseDecoder, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.decoder_3 = base.make_layer(block, nb_filter[2] + nb_filter[3], nb_filter[2])
        self.decoder_2 = base.make_layer(block, nb_filter[1] + nb_filter[2], nb_filter[1])
        self.decoder_1 = base.make_layer(block, nb_filter[0] + nb_filter[1], nb_filter[0])
        self.head = nn.Conv2d(nb_filter[0], num_classes, 1)

    def forward(self, observations):
        z_3 = self.decoder_3(torch.cat([observations[2], self.up(observations[3])], 1))
        z_2 = self.decoder_2(torch.cat([observations[1], self.up(z_3)], 1))
        z_1 = self.decoder_1(torch.cat([observations[0], self.up(z_2)], 1))
        pred_z_1 = self.head(z_1)

        return pred_z_1

class CSBaseDecoder(nn.Module):
    def __init__(self, num_classes, nb_filter):
        super(CSBaseDecoder, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.decoder_3 = CSHDG(nb_filter[2] + nb_filter[3], nb_filter[2])
        self.decoder_2 = CSHDG(nb_filter[1] + nb_filter[2], nb_filter[1])
        self.decoder_1 = CSHDG(nb_filter[0] + nb_filter[1], nb_filter[0])
        self.head = nn.Conv2d(nb_filter[0], num_classes, 1)

    def forward(self, observations):
        z_3 = self.decoder_3(torch.cat([observations[2], self.up(observations[3])], 1))
        z_2 = self.decoder_2(torch.cat([observations[1], self.up(z_3)], 1))
        z_1 = self.decoder_1(torch.cat([observations[0], self.up(z_2)], 1))
        pred_z_1 = self.head(z_1)

        return pred_z_1


#不使用PWConv，直接在内部改变通道
class SSRD(nn.Module):
    def __init__(self, num_classes,  nb_filter):
        super(SSRD, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)#aligh_corners设置为TRUE

        
        self.updecoder_3 = CSHDG(nb_filter[2] + nb_filter[3], nb_filter[2])
        self.updecoder_2 = CSHDG(nb_filter[1] + nb_filter[2], nb_filter[1])
        self.updecoder_1 = CSHDG(nb_filter[0] + nb_filter[1], nb_filter[0])
        
        self.downdecoder_1 = CSHDG(nb_filter[1] + nb_filter[0], nb_filter[1])
        self.downdecoder_2 = CSHDG(nb_filter[2] + nb_filter[1], nb_filter[2])
        

        self.head_1 = nn.Conv2d(nb_filter[0], num_classes, 1, bias=True)
        self.head_2 = nn.Conv2d(nb_filter[1], num_classes, 1, bias=True)
        self.head_3 = nn.Conv2d(nb_filter[2], num_classes, 1, bias=True)

    def forward(self, observations):
        #上采样得到z_3, z_2, z_1
        z_3 = self.updecoder_3(torch.cat([observations[2], self.up(observations[3])], 1))
        z_2 = self.updecoder_2(torch.cat([observations[1], self.up(z_3)], 1))
        z_1 = self.updecoder_1(torch.cat([observations[0], self.up(z_2)], 1))
        pred_z_1 = self.head_1(z_1)

        
        d_2 = self.downdecoder_1(torch.cat([F.interpolate(z_1, scale_factor=0.5, mode="area"), z_2], 1))
        pred_z_2 = self.head_2(d_2)
        d_3 = self.downdecoder_2(torch.cat([F.interpolate(d_2, scale_factor=0.5, mode="area"), z_3], 1))
        pred_z_3 = self.head_3(d_3)


        return [pred_z_1, pred_z_2, pred_z_3]



class CSNet(nn.Module):
    def __init__(self, num_frames=5, num_classes=1, in_channels=3, block=base.ResBlock, num_blocks=[2, 2, 2],
                 nb_filter=[8, 16, 32, 64],  modulator=base.CBAM, use_sufficiency_loss=True, deep_supervision=False):
        super(CSNet, self).__init__()
        self.num_levels = len(nb_filter)    
        self.num_frames = num_frames       
        self.use_sufficiency_loss = use_sufficiency_loss
        self.deep_supervision = deep_supervision

        self.encoder = base.ResNet(in_channels, block, num_blocks, nb_filter)

        self.cshd_blocks = nn.ModuleList([
            CSHDBlock(num_feat=nb_filter[j]) for j in range(self.num_levels)
        ])
        self.msam = MSAM(block, nb_filter, num_frames, self.num_levels)

        self.ssrd = SSRD(num_classes, nb_filter)



    #特征均为4维张量
    def forward(self, x):
        all_level_feats = [[] for _ in range(self.num_levels)]

        for i in range(self.num_frames):#0,1,2,3,4
            frame_feats = self.encoder(x[:, :, i, :, :])
            for j in range(self.num_levels):#0,1,2,3
                all_level_feats[j].append(frame_feats[j])


        all_compensated_feats = [[] for _ in range(self.num_levels)]

        for i in range(self.num_levels):
            for j in range(self.num_frames):  
                all_compensated_feats[i].append(self.cshd_blocks[i](all_level_feats[i][j], all_level_feats[i][-1]))



        #observation[0,1,2,3] = [[1, 8, 384, 384],[1, 16, 192, 192],[1, 32, 96, 96],[1, 64, 48, 48]]
        observations = self.msam([torch.cat(feats, dim=1) for feats in all_compensated_feats])
        
        pred = self.ssrd(observations)

        return pred

if __name__ == '__main__':
    model = CSNet().cuda()
    inputs = torch.randn((1, 3, 5, 384, 384)).cuda()  # Params = 0.57M FLOPs = 10.14G
    flops, params = profile(model, (inputs,))
    print('Params = ' + str(round(params / 1000 ** 2, 2)) + 'M')
    print('FLOPs = ' + str(round(flops / 1000 ** 3, 2)) + 'G')
    flops, params = clever_format([flops, params], '%.6f')
    print('Params = ' + params)
    print('FLOPs = ' + flops)
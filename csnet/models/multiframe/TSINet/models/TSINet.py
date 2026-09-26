import math
from types import SimpleNamespace

import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn

from .MSIT import MSIT
from .Res_block import Res_CBAM_block


def get_CTranS_config():
    config = SimpleNamespace()
    config.transformer = SimpleNamespace()
    config.KV_size = 224  # KV channel dimension size = C1 + C2 + C3 + C4
    config.transformer.num_layers = 3  # the number of SCTB
    config.expand_ratio = 2.66  # CFN channel dimension expand ratio
    config.base_channel = 32  # base channel of SCTransNet
    config.patch_sizes = [16, 8, 4]
    # config.patch_sizes = [32, 16, 8, 4]
    config.n_classes = 1
    return config

class Conv2d_cd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):

        super(Conv2d_cd, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)
        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal
        else:
            # [C_out,C_in, kernel_size,kernel_size] = self.conv.weight.shape
            kernel_diff1 = self.conv.weight.sum(2).sum(2)
            kernel_diff2 = kernel_diff1[:, :, None, None]
            out_diff = F.conv2d(input=x, weight=kernel_diff2, bias=self.conv.bias, stride=self.conv.stride, padding=0,
                                groups=self.conv.groups)
            return out_normal - self.theta * out_diff


class CD_Conv(nn.Module):
    def __init__(self, G0, G):
        super(CD_Conv, self).__init__()
        self.conv = Conv2d_cd(G0, G, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        buffer = self.conv(x)
        output = self.relu(buffer)
        return torch.cat((x, output), dim=1)

class TA(nn.Module):

    def __init__(self, in_dim):
        super(TA, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_dim*2,
            out_channels=2,
            kernel_size=3,
            padding=1
        )
        self.v_t = nn.Parameter(torch.randn((1, in_dim, 1, 1)), requires_grad=True)
        self.v_c = nn.Parameter(torch.randn((1, in_dim, 1, 1)), requires_grad=True)
    def forward(self, t, c):
        attmap = self.conv(torch.cat((t, c), 1))
        attmap = torch.sigmoid(attmap)
        t = attmap[:, 0:1, :, :] * t * self.v_t
        c = attmap[:, 1:, :, :] * c * self.v_c
        out = t + c

        return out

class FIAM(nn.Module):
    def __init__(self, config, n_channels, img_size=256, vis=False, mode='train', deepsuper=True):
        super().__init__()
        self.vis = vis
        self.deepsuper = deepsuper
        print('Deep-Supervision:', deepsuper)
        self.mode = mode
        in_channels = config.base_channel  # 32
        block = Res_CBAM_block
        self.pool = nn.MaxPool2d(2, 2)
        self.inc = self._make_layer(block, n_channels, in_channels)
        self.down_encoder1 = self._make_layer(block, in_channels, in_channels * 2, 1)  # 64  128
        self.down_encoder2 = self._make_layer(block, in_channels * 2, in_channels * 4, 1)  # 64  128
        self.down_encoder3 = self._make_layer(block, in_channels * 4, in_channels * 8, 1)  # 64  128
        self.MSIT = MSIT(config, vis, img_size,
                                       channel_num=[in_channels, in_channels * 2, in_channels * 4],
                                       patchSize=config.patch_sizes)
        self.neck = self._make_layer(block, in_channels * 8, in_channels * 8, 1)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up_8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.decoder3 = self._make_layer(block, in_channels * 12, in_channels * 4, 1)
        self.decoder2 = self._make_layer(block, in_channels * 6, in_channels * 2, 1)
        self.decoder1 = self._make_layer(block, in_channels * 3, in_channels, 1)
        self.final = nn.Conv2d(n_channels, 1, kernel_size=(1, 1), stride=(1, 1))
        if self.deepsuper:
            self.outconv = Res_CBAM_block(15 * in_channels, n_channels)

    def _make_layer(self, block, input_channels, output_channels, num_blocks=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(num_blocks - 1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down_encoder1(self.pool(x1))
        x3 = self.down_encoder2(self.pool(x2))
        d4 = self.down_encoder3(self.pool(x3))

        # Skip Connection
        f1 = x1
        f2 = x2
        f3 = x3

        x1, x2, x3, att_weights = self.MSIT(x1, x2, x3)

        x1 = x1 + f1
        x2 = x2 + f2
        x3 = x3 + f3

        d3 = self.decoder3(torch.cat([self.up(self.neck(d4)), x3], 1))
        d2 = self.decoder2(torch.cat([self.up(d3), x2], 1))
        d1 = self.decoder1(torch.cat([self.up(d2), x1], 1))

        # Deep Supervision
        if self.deepsuper:
            gt4 = self.up_8(d4)
            gt3 = self.up_4(d3)
            gt2 = self.up(d2)
            d0 = self.outconv(torch.cat((gt2, gt3, gt4, d1), 1))

            if self.mode == 'train':
                d0 = self.final(d0)
                return d0
            else:
                out = self.final(d0)
                return out
        else:
            out = self.final(d1)
            return out

class Net(nn.Module):
    def __init__(self, num_frames, nf=16, in_channels=1, block=Res_CBAM_block):
        super(Net, self).__init__()
        self.static_path = block(nf, nf*2)
        self.dynamic_path = block(nf*num_frames, nf*2)
        config_vit = get_CTranS_config()
        self.unet = FIAM(config_vit, nf)
        self.fusion = block(nf*4, nf)
        self.init_conv1 = CD_Conv(in_channels, nf-in_channels)
        self.align1 = TA(in_dim=nf)

    def forward(self, x):
        b, n, c0, h, w = x.shape
        x1 = x[:, -1, :, :, :]
        x1 = self.init_conv1(x1)
        out1 = self.static_path(x1)
        x_m = []
        for i in range(n - 1):
            x_i = x[:, i, :, :, :]
            x_i = self.init_conv1(x_i)
            x_i = self.align1(x_i, x1)
            x_m.append(x_i)
        buffer2 = torch.cat(x_m, dim=1)
        buffer2 = torch.cat((buffer2, x1), dim=1)
        out2 = self.dynamic_path(buffer2)
        buffer = torch.cat((out1, out2), dim=1)
        buffer = self.fusion(buffer)
        out = self.unet(buffer)
        return out

class TSINet(Net):
    """TSINet adapted to the Deep-MIST input layout: (B, C, T, H, W)."""
    def __init__(self, num_frames=5, nf=16, in_channels=3, block=Res_CBAM_block):
        super(TSINet, self).__init__(num_frames=num_frames, nf=nf, in_channels=in_channels, block=block)

    def forward(self, x):
        # x: (B, C, T, H, W) -> (B, T, C, H, W)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        return super(TSINet, self).forward(x)


if __name__ == '__main__':
    from thop import profile, clever_format
    model = TSINet().cuda()
    inputs = torch.randn((1, 3, 5, 384, 384)).cuda()
    flops, params = profile(model, (inputs,))
    print('Params = ' + str(round(params / 1000 ** 2, 2)) + 'M')
    print('FLOPs = ' + str(round(flops / 1000 ** 3, 2)) + 'G')
    flops, params = clever_format([flops, params], '%.6f')
    print('Params = ' + params)
    print('FLOPs = ' + flops)
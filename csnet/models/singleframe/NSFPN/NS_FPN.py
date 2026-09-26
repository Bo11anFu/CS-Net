import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from .diff_cross_attns import SpiralAware_CrossDeformAttn2D


class ConvModule(nn.Module):
    """Simplified ConvModule (Conv -> BN -> ReLU)"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, norm=True, activation=True):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=not norm)]
        if norm:
            layers.append(nn.BatchNorm2d(out_channels))
        if activation:
            layers.append(nn.ReLU(inplace=False))
        self.module = nn.Sequential(*layers)

    def forward(self, x):
        return self.module(x)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7, bn_before_sigmoid=False):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.bn_before_sigmoid = bn_before_sigmoid
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        if bn_before_sigmoid:
            self.bn = nn.BatchNorm2d(1)
            self.bn.bias.data.fill_(0)
            self.bn.bias.requires_grad = False
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        if self.bn_before_sigmoid:
            x = self.bn(x)
        return self.sigmoid(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class LearnableGaussianFilterBank(nn.Module):
    def __init__(self, kernel_size, num_filters, num_channels):
        super(LearnableGaussianFilterBank, self).__init__()
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.C = num_channels
        self.padding = kernel_size // 2
        self.sigmas = nn.ParameterList([nn.Parameter(torch.tensor([1.0])) for _ in range(num_filters)])

    def forward(self, x):
        weights = [self._gaussian_kernel(self.kernel_size, sigma).repeat(self.C, 1, 1, 1) for sigma in self.sigmas]
        filtered_outputs = [F.conv2d(F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode='replicate'),
                                     weight.to(x.device), groups=self.C) for weight in weights]
        return torch.cat(filtered_outputs, dim=1)

    def _gaussian_kernel(self, kernel_size, sigma):
        kernel = torch.zeros(1, 1, kernel_size, kernel_size)
        center = kernel_size // 2
        for i in range(kernel_size):
            for j in range(kernel_size):
                kernel[:, :, i, j] = torch.exp(-((i - center) ** 2 + (j - center) ** 2) / (2 * sigma ** 2))
        return kernel / kernel.sum()


class wav_Enhance(nn.Module):
    """Low-frequency Guided Feature Purification (LFP Module)"""
    def __init__(self, in_channels, wave='haar', mode='symmetric', with_gauss=True, gauss_gate=0.5):
        super(wav_Enhance, self).__init__()
        self.with_gauss = with_gauss
        self.gauss_gate = gauss_gate

        self.attention = SpatialAttention()
        if self.with_gauss:
            self.gaussian_filter = LearnableGaussianFilterBank(kernel_size=3, num_filters=1, num_channels=3 * in_channels)

        self.conv_dwt = nn.Conv2d(in_channels, 4 * in_channels, kernel_size=2, stride=2, bias=False)
        self.conv_idwt = nn.ConvTranspose2d(4 * in_channels, in_channels, kernel_size=2, stride=2, bias=False)

        self._init_dwt_idwt()

    def _init_dwt_idwt(self):
        with torch.no_grad():
            self.conv_dwt.weight.zero_()
            self.conv_idwt.weight.zero_()

    def _haar_dwt(self, x):
        B, C, H, W = x.shape
        x_even = x[:, :, :, 0::2]
        x_odd = x[:, :, :, 1::2]
        LL = (x_even[:, :, 0::2, :] + x_even[:, :, 1::2, :] + x_odd[:, :, 0::2, :] + x_odd[:, :, 1::2, :]) / 2.0
        LH = (x_even[:, :, 0::2, :] - x_even[:, :, 1::2, :] + x_odd[:, :, 0::2, :] - x_odd[:, :, 1::2, :]) / 2.0
        HL = (x_even[:, :, 0::2, :] + x_even[:, :, 1::2, :] - x_odd[:, :, 0::2, :] - x_odd[:, :, 1::2, :]) / 2.0
        HH = (x_even[:, :, 0::2, :] - x_even[:, :, 1::2, :] - x_odd[:, :, 0::2, :] + x_odd[:, :, 1::2, :]) / 2.0
        return LL, torch.cat([LH, HL, HH], dim=1)

    def _haar_idwt(self, LL, Yh):
        B, C, H, W = LL.shape
        LH = Yh[:, :C, :, :]
        HL = Yh[:, C:2*C, :, :]
        HH = Yh[:, 2*C:, :, :]
        x_even_top = (LL + LH + HL + HH) / 2.0
        x_even_bottom = (LL - LH + HL - HH) / 2.0
        x_odd_top = (LL + LH - HL - HH) / 2.0
        x_odd_bottom = (LL - LH - HL + HH) / 2.0
        B, C, H_half, W_half = x_even_top.shape
        out = torch.zeros(B, C, H_half * 2, W_half * 2, device=LL.device, dtype=LL.dtype)
        out[:, :, 0::2, 0::2] = x_even_top
        out[:, :, 0::2, 1::2] = x_odd_top
        out[:, :, 1::2, 0::2] = x_even_bottom
        out[:, :, 1::2, 1::2] = x_odd_bottom
        return out

    def forward(self, x):
        B, C, H, W = x.shape
        LL, Yh_high = self._haar_dwt(x)

        att = self.attention(LL)
        Yh_high = Yh_high * att

        if self.with_gauss:
            Yh_blurred = self.gaussian_filter(Yh_high)
            mask = (Yh_high.abs() < self.gauss_gate).float()
            Yh_high = Yh_high * (1 - mask) + Yh_blurred * mask

        x_rec = self._haar_idwt(LL, Yh_high)
        return x_rec


class NS_FPN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 relu_before_extra_convs=False,
                 upsample_mode='nearest',
                 use_wav_enhance=True,
                 use_crossattn_topdown=True):
        super(NS_FPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.start_level = start_level
        self.relu_before_extra_convs = relu_before_extra_convs
        self.upsample_mode = upsample_mode
        self.relu = nn.ReLU(inplace=False)

        if end_level == -1:
            self.backbone_end_level = self.num_ins
        else:
            self.backbone_end_level = end_level
        assert self.backbone_end_level <= self.num_ins
        self.add_extra_convs = add_extra_convs

        self.use_wav_enhance = use_wav_enhance
        self.use_crossattn_topdown = use_crossattn_topdown

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        self.wavenhance_list = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(in_channels[i], out_channels, 1)
            fpn_conv = ConvModule(out_channels, out_channels, 3, padding=1)

            if self.use_wav_enhance and i < 4:
                wavenhance = wav_Enhance(out_channels, wave='haar', mode='zero')
            else:
                wavenhance = nn.Identity()

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
            self.wavenhance_list.append(wavenhance)

        if self.use_crossattn_topdown:
            self.crossattn_list = nn.ModuleList()
            for i in range(self.start_level, self.backbone_end_level - 1):
                crossattn = SpiralAware_CrossDeformAttn2D(dim=out_channels, n_heads=8, n_points=4)
                self.crossattn_list.append(crossattn)

        extra_levels = num_outs - (self.backbone_end_level - self.start_level)
        if self.add_extra_convs and extra_levels >= 1:
            self.extra_convs = nn.ModuleList()
            for i in range(extra_levels):
                in_c = self.in_channels[self.backbone_end_level - 1] if i == 0 else out_channels
                self.extra_convs.append(
                    ConvModule(in_c, out_channels, 3, stride=2, padding=1)
                )
        else:
            self.extra_convs = None

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        laterals = []
        for i in range(len(inputs) - self.start_level):
            lateral = self.lateral_convs[i](inputs[i + self.start_level])
            lateral = self.wavenhance_list[i](lateral)
            laterals.append(lateral)

        if self.use_crossattn_topdown:
            for i in range(len(laterals) - 1, 0, -1):
                laterals[i - 1] = self.crossattn_list[i - 1](laterals[i - 1], laterals[i])
        else:
            for i in range(len(laterals) - 1, 0, -1):
                upsampled = F.interpolate(laterals[i], size=laterals[i - 1].shape[-2:], mode=self.upsample_mode)
                laterals[i - 1] = laterals[i - 1] + upsampled

        outs = [self.fpn_convs[i](laterals[i]) for i in range(len(laterals))]

        if self.num_outs > len(outs):
            if self.extra_convs is None:
                for _ in range(self.num_outs - len(outs)):
                    outs.append(F.max_pool2d(outs[-1], kernel_size=1, stride=2))
            else:
                x = inputs[self.backbone_end_level - 1] if self.add_extra_convs else outs[-1]
                for conv in self.extra_convs:
                    if self.relu_before_extra_convs:
                        x = self.relu(x)
                    x = conv(x)
                    outs.append(x)

        return outs
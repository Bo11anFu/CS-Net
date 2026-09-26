import math
from types import SimpleNamespace

import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.nn import Dropout, Softmax, Conv2d, LayerNorm
import copy
import numbers
from torch.nn.modules.utils import _pair
from einops import rearrange
from torch.autograd import Variable

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


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class Channel_Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """

    def __init__(self, config, patchsize, img_size, in_channels):
        super().__init__()
        img_size = _pair(img_size)
        patch_size = _pair(patchsize)
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=in_channels,
                                       kernel_size=patch_size,
                                       stride=patch_size)

    def forward(self, x):
        if x is None:
            return None
        x = self.patch_embeddings(x)
        return x


class Reconstruct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(Reconstruct, self).__init__()
        if kernel_size == 3:
            padding = 1
        else:
            padding = 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor

    def forward(self, x):
        if x is None:
            return None
        x = nn.Upsample(scale_factor=self.scale_factor, mode='bilinear')(x)
        out = self.conv(x)
        out = self.norm(out)
        out = self.activation(out)
        return out


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm3d(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm3d, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class SSCA(nn.Module):
    def __init__(self, config, vis, channel_num):
        super(SSCA, self).__init__()
        self.vis = vis
        self.KV_size = config.KV_size
        self.num_attention_heads = 1
        self.psi = nn.InstanceNorm2d(self.num_attention_heads)
        self.softmax = Softmax(dim=3)
        self.q1 = DepthwiseSeparableConv(channel_num[0], channel_num[0] * self.num_attention_heads)
        self.q2 = DepthwiseSeparableConv(channel_num[1], channel_num[1] * self.num_attention_heads)
        self.q3 = DepthwiseSeparableConv(channel_num[2], channel_num[2] * self.num_attention_heads)

        self.k = DepthwiseSeparableConv(self.KV_size, self.KV_size * self.num_attention_heads)
        self.v = DepthwiseSeparableConv(self.KV_size, self.KV_size * self.num_attention_heads)
        self.project_out1 = nn.Conv2d(channel_num[0], channel_num[0], kernel_size=1, bias=False)
        self.project_out2 = nn.Conv2d(channel_num[1], channel_num[1], kernel_size=1, bias=False)
        self.project_out3 = nn.Conv2d(channel_num[2], channel_num[2], kernel_size=1, bias=False)

    def forward(self, emb1, emb2, emb3, emb_all):
        b, c, h, w = emb1.shape

        q1 = self.q1(emb1)
        q2 = self.q2(emb2)
        q3 = self.q3(emb3)
        k = self.k(emb_all)
        v = self.v(emb_all)

        q1 = rearrange(q1, 'b (head c) h w -> b head c (h w)', head=self.num_attention_heads)
        q2 = rearrange(q2, 'b (head c) h w -> b head c (h w)', head=self.num_attention_heads)
        q3 = rearrange(q3, 'b (head c) h w -> b head c (h w)', head=self.num_attention_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_attention_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_attention_heads)

        q1 = torch.nn.functional.normalize(q1, dim=-1)
        q2 = torch.nn.functional.normalize(q2, dim=-1)
        q3 = torch.nn.functional.normalize(q3, dim=-1)

        k = torch.nn.functional.normalize(k, dim=-1)

        _, _, c1, _ = q1.shape
        _, _, c2, _ = q2.shape
        _, _, c3, _ = q3.shape
        _, _, c, _ = k.shape

        # Attention
        attn1 = (q1 @ k.transpose(-2, -1)) / math.sqrt(self.KV_size)
        attn2 = (q2 @ k.transpose(-2, -1)) / math.sqrt(self.KV_size)
        attn3 = (q3 @ k.transpose(-2, -1)) / math.sqrt(self.KV_size)

        attention_probs1 = self.softmax(self.psi(attn1))
        attention_probs2 = self.softmax(self.psi(attn2))
        attention_probs3 = self.softmax(self.psi(attn3))

        out1 = (attention_probs1 @ v)
        out2 = (attention_probs2 @ v)
        out3 = (attention_probs3 @ v)

        out_1 = out1.mean(dim=1)
        out_2 = out2.mean(dim=1)
        out_3 = out3.mean(dim=1)

        out_1 = rearrange(out_1, 'b  c (h w) -> b c h w', h=h, w=w)
        out_2 = rearrange(out_2, 'b  c (h w) -> b c h w', h=h, w=w)
        out_3 = rearrange(out_3, 'b  c (h w) -> b c h w', h=h, w=w)

        O1 = self.project_out1(out_1)
        O2 = self.project_out2(out_2)
        O3 = self.project_out3(out_3)

        weights = None

        return O1, O2, O3, weights


class MSITB(nn.Module):
    def __init__(self, config, vis, channel_num):
        super(MSITB, self).__init__()
        expand_ratio = config.expand_ratio
        self.attn_norm1 = LayerNorm3d(channel_num[0], LayerNorm_type='WithBias')
        self.attn_norm2 = LayerNorm3d(channel_num[1], LayerNorm_type='WithBias')
        self.attn_norm3 = LayerNorm3d(channel_num[2], LayerNorm_type='WithBias')
        self.attn_norm = LayerNorm3d(config.KV_size, LayerNorm_type='WithBias')

        self.channel_attn = SSCA(config, vis, channel_num)

        self.ffn_norm1 = LayerNorm3d(channel_num[0], LayerNorm_type='WithBias')
        self.ffn_norm2 = LayerNorm3d(channel_num[1], LayerNorm_type='WithBias')
        self.ffn_norm3 = LayerNorm3d(channel_num[2], LayerNorm_type='WithBias')

        self.ffn1 = Res_CBAM_block(channel_num[0], channel_num[0])
        self.ffn2 = Res_CBAM_block(channel_num[1], channel_num[1])
        self.ffn3 = Res_CBAM_block(channel_num[2], channel_num[2])

    def forward(self, emb1, emb2, emb3):
        embcat = []
        org1 = emb1
        org2 = emb2
        org3 = emb3
        # org4 = emb4
        for i in range(3):
            var_name = "emb" + str(i + 1)
            tmp_var = locals()[var_name]
            if tmp_var is not None:
                embcat.append(tmp_var)
        emb_all = torch.cat(embcat, dim=1)
        cx1 = self.attn_norm1(emb1) if emb1 is not None else None
        cx2 = self.attn_norm2(emb2) if emb2 is not None else None
        cx3 = self.attn_norm3(emb3) if emb3 is not None else None
        emb_all = self.attn_norm(emb_all)  # 1 196 960
        cx1, cx2, cx3, weights = self.channel_attn(cx1, cx2, cx3, emb_all)
        cx1 = org1 + cx1 if emb1 is not None else None
        cx2 = org2 + cx2 if emb2 is not None else None
        cx3 = org3 + cx3 if emb3 is not None else None

        org1 = cx1
        org2 = cx2
        org3 = cx3

        x1 = self.ffn_norm1(cx1) if emb1 is not None else None
        x2 = self.ffn_norm2(cx2) if emb2 is not None else None
        x3 = self.ffn_norm3(cx3) if emb3 is not None else None

        x1 = self.ffn1(x1) if emb1 is not None else None
        x2 = self.ffn2(x2) if emb2 is not None else None
        x3 = self.ffn3(x3) if emb3 is not None else None

        x1 = x1 + org1 if emb1 is not None else None
        x2 = x2 + org2 if emb2 is not None else None
        x3 = x3 + org3 if emb3 is not None else None

        return x1, x2, x3, weights


class Encoder(nn.Module):
    def __init__(self, config, vis, channel_num):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm1 = LayerNorm3d(channel_num[0], LayerNorm_type='WithBias')
        self.encoder_norm2 = LayerNorm3d(channel_num[1], LayerNorm_type='WithBias')
        self.encoder_norm3 = LayerNorm3d(channel_num[2], LayerNorm_type='WithBias')
        # self.encoder_norm4 = LayerNorm3d(channel_num[3], LayerNorm_type='WithBias')
        for _ in range(config.transformer.num_layers):
            layer = MSITB(config, vis, channel_num)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, emb1, emb2, emb3):
        attn_weights = []
        for layer_block in self.layer:
            emb1, emb2, emb3, weights = layer_block(emb1, emb2, emb3)
            if self.vis:
                attn_weights.append(weights)
        emb1 = self.encoder_norm1(emb1) if emb1 is not None else None
        emb2 = self.encoder_norm2(emb2) if emb2 is not None else None
        emb3 = self.encoder_norm3(emb3) if emb3 is not None else None
        # emb4 = self.encoder_norm4(emb4) if emb4 is not None else None
        return emb1, emb2, emb3, attn_weights


class MSIT(nn.Module):
    def __init__(self, config, vis, img_size, channel_num, patchSize):
        super(MSIT, self).__init__()
        self.patchSize_1 = patchSize[0]
        self.patchSize_2 = patchSize[1]
        self.patchSize_3 = patchSize[2]
        self.embeddings_1 = Channel_Embeddings(config, self.patchSize_1, img_size=img_size, in_channels=channel_num[0])
        self.embeddings_2 = Channel_Embeddings(config, self.patchSize_2, img_size=img_size // 2,
                                               in_channels=channel_num[1])
        self.embeddings_3 = Channel_Embeddings(config, self.patchSize_3, img_size=img_size // 4,
                                               in_channels=channel_num[2])
        self.encoder = Encoder(config, vis, channel_num)

        self.reconstruct_1 = Reconstruct(channel_num[0], channel_num[0], kernel_size=1,
                                         scale_factor=(self.patchSize_1, self.patchSize_1))
        self.reconstruct_2 = Reconstruct(channel_num[1], channel_num[1], kernel_size=1,
                                         scale_factor=(self.patchSize_2, self.patchSize_2))
        self.reconstruct_3 = Reconstruct(channel_num[2], channel_num[2], kernel_size=1,
                                         scale_factor=(self.patchSize_3, self.patchSize_3))

    def forward(self, en1, en2, en3):
        emb1 = self.embeddings_1(en1)
        emb2 = self.embeddings_2(en2)
        emb3 = self.embeddings_3(en3)

        encoded1, encoded2, encoded3, attn_weights = self.encoder(emb1, emb2, emb3)  # (B, n_patch, hidden)

        x1 = self.reconstruct_1(encoded1) if en1 is not None else None
        x2 = self.reconstruct_2(encoded2) if en2 is not None else None
        x3 = self.reconstruct_3(encoded3) if en3 is not None else None

        x1 = x1 + en1 if en1 is not None else None
        x2 = x2 + en2 if en2 is not None else None
        x3 = x3 + en3 if en3 is not None else None

        return x1, x2, x3, attn_weights

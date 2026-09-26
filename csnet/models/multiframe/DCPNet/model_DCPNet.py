"""DCPNet (dual-domain / deformable cross-frame perception) ported from
``MSmsd/models/multiframe/DCPNet-main``. Only the NATTEN import was made
portable: the CUDA extension is used when it is installed, otherwise the
numerically equivalent pure-PyTorch primitives in ``natten_fallback`` are used.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_

try:
    from natten.functional import NATTEN2DQKRPBFunction, NATTEN2DAVFunction
except ImportError:
    try:
        from .natten_fallback import NATTEN2DQKRPBFunction, NATTEN2DAVFunction
    except ImportError:
        from natten_fallback import NATTEN2DQKRPBFunction, NATTEN2DAVFunction

from thop import profile, clever_format


def make_layer(block, in_channels, out_channels, num_blocks=1):
    layers = [block(in_channels, out_channels)]
    for _ in range(num_blocks - 1):
        layers.append(block(out_channels, out_channels))
    return nn.Sequential(*layers)


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


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_channels, ratio=ratio)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x


class MSCAM(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(MSCAM, self).__init__()
        inter_channels = int(in_channels // ratio)
        self.local_att = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
        )
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, inter_channels, 1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        xl = self.local_att(x)
        xg = self.global_att(x)
        xlg = xl + xg
        w = self.sigmoid(xlg)
        return w * x


class Res_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Res_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, in_channels, block, num_blocks, nb_filter):
        super(ResNet, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        # 不下采样，输出通道 = 8，分辨率 = 原始
        self.encoder_0 = make_layer(block, in_channels, nb_filter[0])
        # 每个 stage 前先池化下采样
        self.encoder_1 = make_layer(block, nb_filter[0], nb_filter[1], num_blocks[0])
        self.encoder_2 = make_layer(block, nb_filter[1], nb_filter[2], num_blocks[1])
        self.encoder_3 = make_layer(block, nb_filter[2], nb_filter[3], num_blocks[2])

    def forward(self, x):  # x = [B, C, 384, 384]
        x_e0 = self.encoder_0(x)                  # [B, 8, 384, 384]
        x_e1 = self.encoder_1(self.pool(x_e0))    # [B, 16, 192, 192]
        x_e2 = self.encoder_2(self.pool(x_e1))    # [B, 32, 96, 96]
        x_e3 = self.encoder_3(self.pool(x_e2))    # [B, 64, 48, 48]
        return x_e0, x_e1, x_e2, x_e3


# ======================= [BiM] BiM Gating Block =======================
class BiMGatingBlock(nn.Module):
    """
    使用 BiM 信息 (R, sin(phi), cos(phi)) 对中间特征做调制:
      feat: [B, C, H, W]
      bim:  [B, 3, H, W]  (R, sin(phi), cos(phi))
    """
    def __init__(self, feat_channels: int):
        super().__init__()
        # DEM: 距离 R 的 embedding, 输入 1 通道 -> 输出 feat_channels
        self.dem = nn.Sequential(
            nn.Conv2d(1, feat_channels * 4, 1),
            nn.PReLU(feat_channels * 4),
            nn.Conv2d(feat_channels * 4, feat_channels, 1),
        )
        # AEM: 角度 (sin,cos) 的 embedding, 输入 2 通道 -> 输出 feat_channels
        self.aem = nn.Sequential(
            nn.Conv2d(2, feat_channels * 4, 1),
            nn.PReLU(feat_channels * 4),
            nn.Conv2d(feat_channels * 4, feat_channels, 1),
        )

        # 对 feat、本身的 r_emb、phi_emb 分别做轻量 conv，再用 r*phi 调制
        self.conv_FV = nn.Sequential(
            nn.GroupNorm(8, feat_channels),
            nn.Conv2d(feat_channels, feat_channels, 3, padding=1),
            nn.PReLU(feat_channels),
        )
        self.conv_r = nn.Sequential(
            nn.GroupNorm(8, feat_channels),
            nn.Conv2d(feat_channels, feat_channels, 3, padding=1),
            nn.PReLU(feat_channels),
        )
        self.conv_phi = nn.Sequential(
            nn.GroupNorm(8, feat_channels),
            nn.Conv2d(feat_channels, feat_channels, 3, padding=1),
            nn.PReLU(feat_channels),
        )

    def forward(self, feat: torch.Tensor, bim: torch.Tensor):
        """
        feat: [B, C, H, W]
        bim:  [B, 3, H, W]
        """
        assert bim.shape[1] == 3, "BiM bim should have 3 channels: [R, sin(phi), cos(phi)]"
        r = bim[:, 0:1]       # [B,1,H,W]
        phi = bim[:, 1:3]     # [B,2,H,W]

        r_emb = self.dem(r)        # [B,C,H,W]
        phi_emb = self.aem(phi)    # [B,C,H,W]

        fv = self.conv_FV(feat)
        r_out = self.conv_r(r_emb)
        phi_out = self.conv_phi(phi_emb)

        feat_mod = fv + fv * r_out * phi_out
        return feat_mod
# ======================= [BiM] END =======================


# ======================= [BiM] Content-Aware Upsample =======================
class ContentAwareUpsample(nn.Module):
    """
    简化版 CAUN: 用高分辨率上下文 ctx 预测局部核, 对低分辨率特征 x_low 做内容感知上采样:
      x_low: [B, C, H, W]
      ctx:   [B, C_ctx, H_ctx, W_ctx] (自动对齐到 H,W)
    输出:
      x_up:  [B, C, H*sf, W*sf]
    """
    def __init__(self, in_channels: int, ctx_channels: int, upsample_factor: int = 2):
        super().__init__()
        self.upsample_factor = upsample_factor

        self.ctx_enc = nn.Sequential(
            nn.Conv2d(ctx_channels, ctx_channels, 3, padding=1),
            nn.PReLU(ctx_channels),
            nn.Conv2d(ctx_channels, ctx_channels // 2, 3, padding=1),
            nn.PReLU(ctx_channels // 2),
        )
        # 每个像素预测 C * 9 * sf^2 个权重 (对 C 通道分别做 3x3 convex 插值)
        self.kernel_pred = nn.Conv2d(
            ctx_channels // 2,
            in_channels * 9 * (upsample_factor ** 2),
            1
        )

    def upsample_input(self, inp, mask, upsample_factor):
        """
        inp:  [B, C, H, W]
        mask: [B, C*9*sf*sf, H, W]
        """
        B, C, H, W = inp.shape
        # [B, C, 9, sf, sf, H, W]
        mask = mask.view(B, C, 9, upsample_factor, upsample_factor, H, W)
        mask = torch.softmax(mask, dim=2)

        # 取 3x3 邻域
        inp_pad = F.pad(inp, [1, 1, 1, 1], mode="replicate")
        neigh = F.unfold(inp_pad, kernel_size=3)      # [B, 9*C, H*W]
        neigh = neigh.view(B, C, 9, H, W)             # [B, C, 9, H, W]
        neigh = neigh.unsqueeze(3).unsqueeze(4)       # [B, C, 9, 1, 1, H, W]

        out = (mask * neigh).sum(dim=2)               # [B, C, sf, sf, H, W]
        out = out.permute(0, 1, 5, 3, 4, 2)           # [B, C, H, sf, W, sf]
        out = out.reshape(B, C, H * upsample_factor, W * upsample_factor)
        return out

    def forward(self, x_low: torch.Tensor, ctx: torch.Tensor):
        B, C, H, W = x_low.shape

        # 对齐上下文分辨率
        if ctx.shape[-2:] != (H, W):
            ctx = F.adaptive_avg_pool2d(ctx, (H, W))

        ctx_feat = self.ctx_enc(ctx)
        mask = self.kernel_pred(ctx_feat)  # [B, C*9*sf*sf, H, W]

        x_up = self.upsample_input(x_low, mask, self.upsample_factor)
        return x_up
# ======================= [BiM] END =======================


# ======================= STCP 相关模块 =======================
class DeformableLocalCorrAggr(nn.Module):
    def __init__(self, block, in_channels, neighbor_sizes=[3, 5, 7], dilation_rates=[1, 1, 1]):
        super(DeformableLocalCorrAggr, self).__init__()
        self.in_channels = in_channels

        assert len(neighbor_sizes) == len(dilation_rates), \
            f'Number of neighbor sizes must be equal to number of dilation rates.'
        self.num_levels = len(neighbor_sizes)

        for neighbor_size in neighbor_sizes:
            assert neighbor_size > 1 and neighbor_size % 2 == 1, \
                f'Neighbor size must be an odd number greater than 1, got {neighbor_size}.'
            assert neighbor_size in [3, 5, 7, 9, 11, 13], \
                f'CUDA kernel only supports neighbor sizes 3, 5, 7, 9, 11, and 13; got {neighbor_size}.'
        self.neighbor_sizes = neighbor_sizes

        for dilation_rate in dilation_rates:
            assert dilation_rate >= 1, f'Dilation rate must be >= 1, got {dilation_rate}.'
        self.dilation_rates = dilation_rates

        for i in range(self.num_levels):
            rpb = nn.Parameter(torch.zeros(1, (2 * neighbor_sizes[i] - 1), (2 * neighbor_sizes[i] - 1)))
            trunc_normal_(rpb, mean=0., std=.02, a=-2., b=2.)
            self.register_parameter('rpb{}'.format(i), rpb)

        self.conv_v_list = nn.ModuleList()
        for i in range(self.num_levels):
            conv_v = nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels,
                               kernel_size=1, stride=1, padding=0)
            self.conv_v_list.append(conv_v)

        self.conv_fusion = make_layer(block, self.in_channels * (self.num_levels + 1), self.in_channels, num_blocks=2)

    def forward(self, sup_frame_xi_0, key_frame_xi_0):
        b, _, h, w = key_frame_xi_0.shape  # (b, c, h, w)
        Q = key_frame_xi_0.unsqueeze(1).permute(0, 1, 3, 4, 2)  # (b, 1, h, w, c)
        K = sup_frame_xi_0.unsqueeze(1).permute(0, 1, 3, 4, 2)  # (b, 1, h, w, c)

        all_aggregated_xi_0 = [sup_frame_xi_0]
        for i in range(self.num_levels):
            V = self.conv_v_list[i](sup_frame_xi_0).unsqueeze(1).permute(0, 1, 3, 4, 2)  # (b, 1, h, w, c)
            cv = NATTEN2DQKRPBFunction.apply(Q, K, getattr(self, 'rpb{}'.format(i)), self.neighbor_sizes[i],
                                             self.dilation_rates[i])  # (b, 1, h, w, n*n)
            norm_cv = cv.softmax(dim=-1)
            aggregated_xi_0 = NATTEN2DAVFunction.apply(norm_cv, V, self.neighbor_sizes[i],
                                                       self.dilation_rates[i])  # (b, 1, h, w, c)
            aggregated_xi_0 = aggregated_xi_0.permute(0, 1, 4, 2, 3).contiguous()
            aggregated_xi_0 = aggregated_xi_0.view(b, self.in_channels, h, w)
            all_aggregated_xi_0.append(aggregated_xi_0)

        all_aggregated_xi_0 = torch.cat(all_aggregated_xi_0, dim=1)  # (b, (l+1)*c, h, w)
        aggregated_xi_0 = self.conv_fusion(all_aggregated_xi_0)      # (b, c, h, w)
        return aggregated_xi_0


class IFOffsetWarpAttn(nn.Module):
    """
    纯 PyTorch 实现的可变形局部跨帧注意力：
      1) 预测逐像素 2D offset（dx, dy）
      2) 用 grid_sample 把 support 特征按 offset warp 到 key 上
      3) 用 NATTEN 做局部 cross-attention（可选）
    """
    def __init__(self, in_channels, k_list=(3, 5, 7), dilation_list=(1, 1, 1), use_natten=True):
        super().__init__()
        self.use_natten = use_natten
        self.in_channels = in_channels

        # offset 预测
        self.offset_conv = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels // 2, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, 2, 3, padding=1, bias=False)  # dx, dy
        )

        if use_natten:
            assert len(k_list) == len(dilation_list)
            self.num_levels = len(k_list)
            self.neighbor_sizes = k_list
            self.dilation_rates = dilation_list

            for i in range(self.num_levels):
                rpb = nn.Parameter(torch.zeros(1, (2 * k_list[i] - 1), (2 * k_list[i] - 1)))
                trunc_normal_(rpb, mean=0., std=.02, a=-2., b=2.)
                self.register_parameter(f"rpb{i}", rpb)

            self.conv_v_list = nn.ModuleList([
                nn.Conv2d(in_channels, in_channels, 1, bias=False) for _ in range(self.num_levels)
            ])
            self.fuse = make_layer(Res_block, in_channels * (self.num_levels + 1), in_channels, num_blocks=1)

    @staticmethod
    def _make_base_grid(h, w, device):
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, h, device=device),
            torch.linspace(-1, 1, w, device=device),
            indexing="ij"
        )
        base_grid = torch.stack([xx, yy], dim=-1)  # [H,W,2]
        return base_grid

    def forward(self, sup, key):
        """
        sup,key: [B,C,H,W]
        返回:     [B,C,H,W] (对齐到 key 的 support 特征)
        """
        b, c, h, w = key.shape

        inp = torch.cat([key, sup, key - sup], dim=1)  # [B,3C,H,W]
        offset = self.offset_conv(inp)                 # [B,2,H,W]

        offset_x = 2.0 * offset[:, 0] / max(w - 1, 1)
        offset_y = 2.0 * offset[:, 1] / max(h - 1, 1)
        offset_norm = torch.stack([offset_x, offset_y], dim=-1)  # [B,H,W,2]

        base_grid = self._make_base_grid(h, w, key.device)       # [H,W,2]
        grid = base_grid.unsqueeze(0) + offset_norm              # [B,H,W,2]
        sup_warp = F.grid_sample(sup, grid, mode='bilinear', padding_mode='border', align_corners=True)

        if not self.use_natten:
            return sup_warp

        Q = key.unsqueeze(1).permute(0, 1, 3, 4, 2)        # (b,1,h,w,c)
        K = sup_warp.unsqueeze(1).permute(0, 1, 3, 4, 2)   # (b,1,h,w,c)

        feats = [sup_warp]
        for i in range(self.num_levels):
            V = self.conv_v_list[i](sup_warp).unsqueeze(1).permute(0, 1, 3, 4, 2)  # (b,1,h,w,c)

            cv = NATTEN2DQKRPBFunction.apply(Q, K, getattr(self, f"rpb{i}"),
                                             self.neighbor_sizes[i], self.dilation_rates[i])  # (b,1,h,w,n*n)
            norm_cv = cv.softmax(dim=-1)

            agg = NATTEN2DAVFunction.apply(norm_cv, V,
                                           self.neighbor_sizes[i], self.dilation_rates[i])  # (b,1,h,w,c)
            agg = agg.permute(0, 1, 4, 2, 3).contiguous().view(b, c, h, w)
            feats.append(agg)

        feat_cat = torch.cat(feats, dim=1)  # [B,(L+1)*C,H,W]
        out = self.fuse(feat_cat)
        return out


class SpatialTemporalCorrPyramid(nn.Module):
    def __init__(self, nb_filter, use_natten=True):
        super().__init__()
        self.blocks = nn.ModuleList([
            IFOffsetWarpAttn(nb_filter[0], k_list=(3, 5, 7), use_natten=use_natten),
            IFOffsetWarpAttn(nb_filter[1], k_list=(3, 5, 7), use_natten=use_natten),
            IFOffsetWarpAttn(nb_filter[2], k_list=(3, 5, 7), use_natten=use_natten),
            IFOffsetWarpAttn(nb_filter[3], k_list=(3, 5, 7), use_natten=use_natten),
        ])

    def forward(self, sup_feats, key_feats):
        outs = []
        for i in range(4):
            outs.append(self.blocks[i](sup_feats[i], key_feats[i]))
        return outs


class InterFrameFusion(nn.Module):
    def __init__(self, block, nb_filter, num_inputs, num_stages):
        super(InterFrameFusion, self).__init__()
        self.num_stages = num_stages
        self.inter_frame_fusion_list = nn.ModuleList()
        for i in range(num_stages):
            inter_frame_fusion = make_layer(block, nb_filter[i] * num_inputs, nb_filter[i], num_blocks=1)
            self.inter_frame_fusion_list.append(inter_frame_fusion)

    def forward(self, all_frames_aggregated_x_e):
        fused_x_e = []
        for i in range(self.num_stages):
            fused_x_ei = self.inter_frame_fusion_list[i](all_frames_aggregated_x_e[i])
            fused_x_e.append(fused_x_ei)
        return fused_x_e
# ======================= STCP END =======================


# ======================= [DMM] MemoryBank2D =======================
class MemoryBank2D(nn.Module):
    """
    简单版 Decoder Memory Bank：
      - 输入特征 [B,C,H,W] -> B*H*W 个 patch
      - 与 num_slots 个 memory 向量做余弦相似度 + softmax
      - 用 memory 重构特征，得到 recon
      - 输出：
          recon: [B,C,H,W]
          match_map: [B,H,W] 每个位置的 max softmax 权重（匹配程度）
    """
    def __init__(self, num_slots: int, dim: int, momentum: float = 0.1):
        super().__init__()
        self.num_slots = num_slots
        self.dim = dim
        self.momentum = momentum

        mem = torch.randn(num_slots, dim)
        mem = F.normalize(mem, dim=1)
        # 用 buffer 存 memory，手动更新
        self.register_buffer("memory", mem)

    def forward(self, x: torch.Tensor, update: bool = False):
        """
        x: [B,C,H,W]
        update: 训练时 True，推理时 False
        """
        B, C, H, W = x.shape
        feat = x.permute(0, 2, 3, 1).contiguous().view(-1, C)  # [BHW,C]
        feat_norm = F.normalize(feat, dim=1)

        mem = self.memory                               # [N,C]
        mem_norm = F.normalize(mem, dim=1)

        # 余弦相似度 + softmax
        sim = torch.matmul(feat_norm, mem_norm.t())     # [BHW,N]
        attn = F.softmax(sim, dim=1)                    # [BHW,N]

        # 用 memory 重构
        recon = torch.matmul(attn, mem_norm)            # [BHW,C]
        recon = recon.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()  # [B,C,H,W]

        # 每个位置的最大权重（匹配程度）
        max_attn, _ = attn.max(dim=1)                   # [BHW]
        match_map = max_attn.view(B, H, W)              # [B,H,W]

        # 简单 EMA 更新 memory（top-1 分配）
        if update:
            with torch.no_grad():
                assign = attn.max(dim=1)[1]             # [BHW]
                for n in range(self.num_slots):
                    mask = (assign == n)
                    if mask.any():
                        new_vec = feat_norm[mask].mean(dim=0)
                        new_vec = F.normalize(new_vec, dim=0)
                        old_vec = mem[n]
                        mem[n] = F.normalize(
                            (1.0 - self.momentum) * old_vec + self.momentum * new_vec,
                            dim=0
                        )
                self.memory.copy_(mem)

        return recon, match_map
# ======================= [DMM] END =======================


# ======================= Decoder: BiM + DMM 一起挂 =======================
class BaseDecoder(nn.Module):
    def __init__(self, num_classes, block, nb_filter,
                 use_memory: bool = True, mem_slots: int = 64, mem_momentum: float = 0.1):
        super(BaseDecoder, self).__init__()

        # ===== [BiM] CAUN 上采样结构 =====
        self.caun3 = ContentAwareUpsample(in_channels=nb_filter[3],
                                          ctx_channels=nb_filter[2],
                                          upsample_factor=2)  # 48->96
        self.caun2 = ContentAwareUpsample(in_channels=nb_filter[2],
                                          ctx_channels=nb_filter[1],
                                          upsample_factor=2)  # 96->192
        self.caun1 = ContentAwareUpsample(in_channels=nb_filter[1],
                                          ctx_channels=nb_filter[0],
                                          upsample_factor=2)  # 192->384

        self.conv2_1 = make_layer(block, nb_filter[2] + nb_filter[3], nb_filter[2])
        self.conv1_2 = make_layer(block, nb_filter[1] + nb_filter[2], nb_filter[1])
        self.conv0_3 = make_layer(block, nb_filter[0] + nb_filter[1], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

        # ===== [DMM] Decoder MemoryBank 挂在 x0_3 上 =====
        self.use_memory = use_memory
        if self.use_memory:
            self.memory_low = MemoryBank2D(num_slots=mem_slots,
                                           dim=nb_filter[0],
                                           momentum=mem_momentum)
        # ===== [DMM] END =====

    def forward(self, x0_0, x1_0, x2_0, x3_0, return_mem: bool = False):
        """
        return_mem:
          False: 默认行为，只返回 segmentation logits
          True:  同时返回 memory 信息 dict
        """
        # stage3 -> stage2
        x3_up = self.caun3(x3_0, ctx=x2_0)  # 48->96
        x2_1 = self.conv2_1(torch.cat([x2_0, x3_up], 1))

        # stage2 -> stage1
        x2_up = self.caun2(x2_1, ctx=x1_0)  # 96->192
        x1_2 = self.conv1_2(torch.cat([x1_0, x2_up], 1))

        # stage1 -> stage0
        x1_up = self.caun1(x1_2, ctx=x0_0)  # 192->384
        x0_3 = self.conv0_3(torch.cat([x0_0, x1_up], 1))

        mem_info = None
        if self.use_memory:
            # 训练时更新 memory，eval 时只用但不更新
            recon, match_map = self.memory_low(x0_3, update=self.training)
            # 残差式融合：原特征 + memory 重构
            x0_3 = x0_3 + recon

            if return_mem:
                # 一个简单的“异常分数”：1 - 平均匹配度
                mem_score = 1.0 - match_map.mean(dim=(1, 2))  # [B]
                mem_info = {
                    "match_map": match_map,   # [B,H,W]
                    "mem_score": mem_score    # [B]
                }

        output = self.final(x0_3)

        if return_mem:
            return output, mem_info
        return output
# ======================= Decoder END =======================


class DCPNet(nn.Module):
    def __init__(self, num_inputs=5, num_classes=1, in_channels=3, block=Res_block,
                 num_blocks=[2, 2, 2, 2], nb_filter=[8, 16, 32, 64, 128], num_stages=4):
        super(DCPNet, self).__init__()
        self.num_inputs = num_inputs

        self.Encoder = ResNet(in_channels, block, num_blocks, nb_filter)

        # [BiM] 是否启用 BiM gating（可以在外面改 self.use_bim 做 ablation）
        self.use_bim = True
        self.bim_gates = nn.ModuleList([
            BiMGatingBlock(nb_filter[0]),
            BiMGatingBlock(nb_filter[1]),
            BiMGatingBlock(nb_filter[2]),
            BiMGatingBlock(nb_filter[3]),
        ])

        self.STCP = SpatialTemporalCorrPyramid(nb_filter, use_natten=True)
        self.IFF = InterFrameFusion(block, nb_filter, num_inputs, num_stages)

        # [DMM] 使用带 memory 的 Decoder
        self.Decoder = BaseDecoder(num_classes, block, nb_filter,
                                   use_memory=True, mem_slots=64, mem_momentum=0.1)

    def forward(self, x, bim: torch.Tensor = None, return_mem: bool = False):
        """
        x:   [B, C, T, H, W]
        bim: [B, 3, H, W] or None
             - 不为 None 时，在 Encoder 的 key frame 各个 stage 做 BiM gating
        return_mem:
             - False: 返回 seg logits
             - True:  返回 (seg logits, mem_info)
        """
        # 关键帧：最后一帧
        key_frame_x_e = self.Encoder(x[:, :, -1, :, :])  # tuple(len=4): (x_e0,x_e1,x_e2,x_e3)

        # ===== [BiM] 在各个 stage 对 key frame 特征做 BiM gating =====
        if self.use_bim and (bim is not None):
            key_feats = list(key_frame_x_e)
            for i in range(4):
                feat_i = key_feats[i]
                # BiM 下采到当前 stage 分辨率
                bim_i = F.interpolate(bim, size=feat_i.shape[-2:], mode='bilinear', align_corners=False)
                key_feats[i] = self.bim_gates[i](feat_i, bim_i)
            key_frame_x_e = tuple(key_feats)
        # ===== [BiM] END =====

        all_frames_aggregated_x_e0 = []
        all_frames_aggregated_x_e1 = []
        all_frames_aggregated_x_e2 = []
        all_frames_aggregated_x_e3 = []
        for i in range(self.num_inputs):
            # 最后一帧用已 gating 的 key_frame_x_e，其余帧单独 encode
            sup_frame_x_e = self.Encoder(x[:, :, i, :, :]) if (i != self.num_inputs - 1) else key_frame_x_e

            aggregated_x_e = self.STCP(sup_frame_x_e, key_frame_x_e)
            all_frames_aggregated_x_e0.append(aggregated_x_e[0])
            all_frames_aggregated_x_e1.append(aggregated_x_e[1])
            all_frames_aggregated_x_e2.append(aggregated_x_e[2])
            all_frames_aggregated_x_e3.append(aggregated_x_e[3])

        # Inter-frame Fusion
        all_frames_aggregated_x_e0 = torch.cat(all_frames_aggregated_x_e0, dim=1)
        all_frames_aggregated_x_e1 = torch.cat(all_frames_aggregated_x_e1, dim=1)
        all_frames_aggregated_x_e2 = torch.cat(all_frames_aggregated_x_e2, dim=1)
        all_frames_aggregated_x_e3 = torch.cat(all_frames_aggregated_x_e3, dim=1)
        fused_x_e = self.IFF([all_frames_aggregated_x_e0, all_frames_aggregated_x_e1,
                              all_frames_aggregated_x_e2, all_frames_aggregated_x_e3])

        preds = self.Decoder(*fused_x_e, return_mem=return_mem)
        return preds


if __name__ == '__main__':
    model = DCPNet().cuda()
    inputs = torch.randn((1, 3, 5, 384, 384)).cuda()   # [B,C,T,H,W]
    flops, params = profile(model, (inputs,))
    print('Params = ' + str(round(params / 1000 ** 2, 2)) + 'M')
    print('FLOPs = ' + str(round(flops / 1000 ** 3, 2)) + 'G')
    flops, params = clever_format([flops, params], '%.6f')
    print('Params = ' + params)
    print('FLOPs = ' + flops)

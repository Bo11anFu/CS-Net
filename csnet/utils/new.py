import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_
from natten.functional import NATTEN2DQKRPBFunction, NATTEN2DAVFunction
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
        self.encoder_0 = make_layer(block, in_channels, nb_filter[0])  # 2
        self.encoder_1 = make_layer(block, nb_filter[0], nb_filter[1], num_blocks[0])  # 4
        self.encoder_2 = make_layer(block, nb_filter[1], nb_filter[2], num_blocks[1])  # 4
        self.encoder_3 = make_layer(block, nb_filter[2], nb_filter[3], num_blocks[2])  # 4

    def forward(self, x):
        x_e0 = self.encoder_0(x)  # [1, 8, 384, 384]
        x_e1 = self.encoder_1(self.pool(x_e0))  # [1, 16, 192, 192]
        x_e2 = self.encoder_2(self.pool(x_e1))  # [1, 32, 96, 96]
        x_e3 = self.encoder_3(self.pool(x_e2))  # [1, 64, 48, 48]

        return x_e0, x_e1, x_e2, x_e3


class SpatialShiftCorrAggr(nn.Module):
    def __init__(self, block, in_channels, shift_size=3, neighbor_sizes=[3, 5, 7], dilation_rates=[1, 1, 1]):
        super(SpatialShiftCorrAggr, self).__init__()
        self.in_channels = in_channels

        self.shift_size = shift_size
        if shift_size != 0:
            self.padding = shift_size
            self.shifts = [(sh, sw) for sh in [-shift_size, 0, shift_size] for sw in [-shift_size, 0, shift_size]
                           if not (sh == 0 and sw == 0)]
            self.num_groups = len(self.shifts)
            self.conv_smooth = nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels,
                                         kernel_size=3, stride=1, padding=1, groups=self.num_groups)
            self.conv_reduce = nn.Conv2d(in_channels=self.in_channels * 2, out_channels=self.in_channels,
                                         kernel_size=1, stride=1, padding=0)

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
            assert dilation_rate >= 1, f'Dilation rate must be greater than or equal to 1, got {dilation_rate}.'
        self.dilation_rates = dilation_rates

        self.conv_v_list = nn.ModuleList()
        for i in range(self.num_levels):
            conv_v = nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels,
                               kernel_size=1, stride=1, padding=0)
            self.conv_v_list.append(conv_v)

            rpb = nn.Parameter(torch.zeros(1, (2 * neighbor_sizes[i] - 1), (2 * neighbor_sizes[i] - 1)))
            trunc_normal_(rpb, mean=0., std=.02, a=-2., b=2.)
            self.register_parameter('rpb{}'.format(i), rpb)

        self.conv_fusion = make_layer(block, self.in_channels * (self.num_levels + 1), self.in_channels,
                                      num_blocks=1)  # 2

    def forward(self, sup_frame_x_ei, key_frame_x_ei):
        b, _, h, w = key_frame_x_ei.shape  # (b, c, h, w)
        all_aggregated_x_ei = [sup_frame_x_ei]

        # ------------------ grouped spatial shift ------------------ #
        if self.shift_size != 0:
            # pad the original support frame feature with zero
            p = self.padding
            pad_sup_frame_x_ei = F.pad(sup_frame_x_ei, (p, p, p, p), "constant", 0)  # (b, c, h+2p, w+2p)
            # chunk the support frame feature into a specified number of groups along the channel dimension
            sup_frame_x_ei_parts = torch.chunk(pad_sup_frame_x_ei, self.num_groups, dim=1)
            # perform grouped spatial shift
            shift_parts = [torch.roll(sup_frame_x_ei_parts[i], shifts=self.shifts[i], dims=(2, 3))
                           for i in range(self.num_groups)]
            # concatenate shifted parts
            shift_sup_frame_x_ei = torch.cat(shift_parts, dim=1)  # (b, c, h+2p, w+2p)
            # crop the shifted feature
            shift_sup_frame_x_ei = shift_sup_frame_x_ei[..., p:-p, p:-p]  # (b, c, h, w)
            # conv smooth
            shift_sup_frame_x_ei = self.conv_smooth(shift_sup_frame_x_ei)  # (b, c, h, w)
            # concatenate the original support frame feature and the shifted feature
            sup_frame_x_ei = torch.cat([sup_frame_x_ei, shift_sup_frame_x_ei], dim=1)  # (b, 2c, h, w)
            # channel reduction
            sup_frame_x_ei = self.conv_reduce(sup_frame_x_ei)  # (b, c, h, w)

        # ------------------ local cross-attention ------------------ #
        # init Q
        Q = key_frame_x_ei.unsqueeze(1).permute(0, 1, 3, 4, 2)  # (b, c, h, w) -> (b, 1, c, h, w) -> (b, 1, h, w, c)
        # init K
        K = sup_frame_x_ei.unsqueeze(1).permute(0, 1, 3, 4, 2)  # (b, c, h, w) -> (b, 1, c, h, w) -> (b, 1, h, w, c)

        for i in range(self.num_levels):
            # update V
            V = self.conv_v_list[i](sup_frame_x_ei).unsqueeze(1).permute(0, 1, 3, 4, 2)  # (b, 1, h, w, c)
            # compute correlation volume
            cv = NATTEN2DQKRPBFunction.apply(Q, K, getattr(self, 'rpb{}'.format(i)), self.neighbor_sizes[i],
                                             self.dilation_rates[i])  # (b, 1, h, w, n*n)
            norm_cv = cv.softmax(dim=-1)  # (b, 1, h, w, n*n)
            # cross-attention
            aggregated_x_ei = NATTEN2DAVFunction.apply(norm_cv, V, self.neighbor_sizes[i],
                                                       self.dilation_rates[i])  # (b, 1, h, w, c)
            aggregated_x_ei = aggregated_x_ei.permute(0, 1, 4, 2, 3).contiguous()  # (b, 1, c, h, w)
            aggregated_x_ei = aggregated_x_ei.view(b, self.in_channels, h, w)  # (b, c, h, w)
            all_aggregated_x_ei.append(aggregated_x_ei)

        # concatenates all aggregated features from each correlation volume pyramid level along the channel dimension
        all_aggregated_x_ei = torch.cat(all_aggregated_x_ei, dim=1)  # (b, (l+1)*c, h, w)
        aggregated_x_ei = self.conv_fusion(all_aggregated_x_ei)  # (b, c, h, w)

        return aggregated_x_ei


class SpatialShiftCorrPyramid(nn.Module):
    def __init__(self, block, nb_filter, num_stages, shift_sizes=[3, 3, 3, 3]):
        super(SpatialShiftCorrPyramid, self).__init__()
        self.num_stages = num_stages
        self.SSCA_list = nn.ModuleList()
        for i in range(num_stages):
            SSCA = SpatialShiftCorrAggr(block, nb_filter[i], shift_size=shift_sizes[i],
                                        neighbor_sizes=[3, 5, 7], dilation_rates=[1, 1, 1])
            self.SSCA_list.append(SSCA)

    def forward(self, sup_frame_x_e, key_frame_x_e):
        aggregated_x_e = []
        for i in range(self.num_stages):
            aggregated_x_ei = self.SSCA_list[i](sup_frame_x_e[i], key_frame_x_e[i])
            aggregated_x_e.append(aggregated_x_ei)

        return aggregated_x_e


class InterFrameFusion(nn.Module):
    def __init__(self, block, nb_filter, num_inputs, num_stages):
        super(InterFrameFusion, self).__init__()
        self.num_stages = num_stages
        self.inter_frame_fusion_list = nn.ModuleList()
        for i in range(num_stages):
            inter_frame_fusion = make_layer(block, nb_filter[i] * num_inputs, nb_filter[i], num_blocks=1)  # 2
            self.inter_frame_fusion_list.append(inter_frame_fusion)

    def forward(self, all_frames_aggregated_x_e):
        fused_x_e = []
        for i in range(self.num_stages):
            fused_x_ei = self.inter_frame_fusion_list[i](all_frames_aggregated_x_e[i])
            fused_x_e.append(fused_x_ei)

        return fused_x_e


class InfoBottleneck(nn.Module):
    def __init__(self, block, in_channels, out_channels, ratio, activation):
        super(InfoBottleneck, self).__init__()
        if activation == 'CBAM':
            self.activation = CBAM(in_channels, ratio=ratio)
        elif activation == 'MS-CAM':
            self.activation = MSCAM(in_channels, ratio=ratio)
        else:
            NotImplementedError(f"Invalid activation type '{activation}'.")
        # self.filter = nn.Conv2d(in_channels, out_channels, 1)
        self.filter = make_layer(block, in_channels, out_channels, num_blocks=1)  # 2

    def forward(self, v):
        z = self.activation(v)
        z = self.filter(z)

        return z


class VSD(nn.Module):
    def __init__(self, block, in_channels, out_channels, num_classes, ratio, activation):
        super(VSD, self).__init__()
        self.bottleneck = InfoBottleneck(block, in_channels, out_channels, ratio, activation)
        self.head_v = nn.Conv2d(in_channels, num_classes, 1)
        self.head_z = nn.Conv2d(out_channels, num_classes, 1)

    def forward(self, v):
        # p(y|v)
        pred_v = self.head_v(v)
        # Variational Information Bottleneck: v -> z
        z = self.bottleneck(v)
        # p(y|z)
        pred_z = self.head_z(z)

        return z, pred_z, pred_v


class ProgressiveDistillationDecoder(nn.Module):
    def __init__(self, num_classes, block, nb_filter, ratio=16, activation='CBAM'):
        super(ProgressiveDistillationDecoder, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

        self.decoder_2 = VSD(block, nb_filter[2] + nb_filter[3], nb_filter[2], num_classes, ratio, activation)
        self.decoder_1 = VSD(block, nb_filter[1] + nb_filter[2], nb_filter[1], num_classes, ratio, activation)
        self.decoder_0 = VSD(block, nb_filter[0] + nb_filter[1], nb_filter[0], num_classes, ratio, activation)

    def forward(self, x_e0, x_e1, x_e2, x_e3):
        # stage 2
        z_d2, pred_z_d2, pred_v_d2 = self.decoder_2(torch.cat([x_e2, self.up(x_e3)], 1))  # [96, 96]
        pred_z_d2, pred_v_d2 = self.up_4(pred_z_d2), self.up_4(pred_v_d2)  # [384, 384]
        # stage 1
        z_d1, pred_z_d1, pred_v_d1 = self.decoder_1(torch.cat([x_e1, self.up(z_d2)], 1))  # [192, 192]
        pred_z_d1, pred_v_d1 = self.up(pred_z_d1), self.up(pred_v_d1)  # [384, 384]
        # stage 0
        z_d0, pred_z_d0, pred_v_d0 = self.decoder_0(torch.cat([x_e0, self.up(z_d1)], 1))  # [384, 384]

        pred = pred_z_d0
        pred_z_list = [pred_z_d0, pred_z_d1, pred_z_d2]
        pred_v_list = [pred_v_d0, pred_v_d1, pred_v_d2]

        return pred, pred_z_list, pred_v_list


class MISTNet(nn.Module):
    def __init__(self, num_inputs=5, num_classes=1, in_channels=3, block=Res_block, num_blocks=[2, 2, 2, 2],
                 nb_filter=[8, 16, 32, 64, 128], num_stages=4, use_ib_loss=True, deep_supervision=False):
        # block=Res_CBAM_block / nb_filter=[16, 32, 64, 128, 256]  [8, 16, 32, 64, 128]
        super(MISTNet, self).__init__()
        self.num_inputs = num_inputs

        # self.Encoder = efficientvit_b1()
        self.Encoder = ResNet(in_channels, block, num_blocks, nb_filter)

        self.SSCP = SpatialShiftCorrPyramid(block, nb_filter, num_stages, shift_sizes=[3, 3, 3, 3])

        self.IFF = InterFrameFusion(block, nb_filter, num_inputs, num_stages)

        self.Decoder = ProgressiveDistillationDecoder(num_classes, block, nb_filter, ratio=16, activation='CBAM')
        # self.Decoder = BABDecoder(num_classes, nb_filter, use_ib_loss, deep_supervision)
        # self.Decoder = BaseDecoder(num_classes, block, nb_filter, deep_supervision)
        # self.Decoder = DenseNestedDecoder(num_classes, block, num_blocks, nb_filter, deep_supervision)

        self.use_ib_loss = use_ib_loss
        if not self.use_ib_loss:
            self.deep_supervision = deep_supervision

    def forward(self, x):
        # key frame feature
        key_frame_x_e = self.Encoder(x[:, :, -1, :, :])
        all_frames_aggregated_x_e0 = []
        all_frames_aggregated_x_e1 = []
        all_frames_aggregated_x_e2 = []
        all_frames_aggregated_x_e3 = []
        for i in range(self.num_inputs):
            # sup frame feature
            sup_frame_x_e = self.Encoder(x[:, :, i, :, :]) if (i != self.num_inputs - 1) else key_frame_x_e

            # -------------------- Spatial-shift Temporal Correlation Pyramid -------------------- #
            aggregated_x_e = self.SSCP(sup_frame_x_e, key_frame_x_e)
            all_frames_aggregated_x_e0.append(aggregated_x_e[0])
            all_frames_aggregated_x_e1.append(aggregated_x_e[1])
            all_frames_aggregated_x_e2.append(aggregated_x_e[2])
            all_frames_aggregated_x_e3.append(aggregated_x_e[3])

        # -------------------- Inter-frame Fusion -------------------- #
        all_frames_aggregated_x_e0 = torch.cat(all_frames_aggregated_x_e0, dim=1)
        all_frames_aggregated_x_e1 = torch.cat(all_frames_aggregated_x_e1, dim=1)
        all_frames_aggregated_x_e2 = torch.cat(all_frames_aggregated_x_e2, dim=1)
        all_frames_aggregated_x_e3 = torch.cat(all_frames_aggregated_x_e3, dim=1)
        fused_x_e = self.IFF([all_frames_aggregated_x_e0, all_frames_aggregated_x_e1,
                              all_frames_aggregated_x_e2, all_frames_aggregated_x_e3])

        # -------------------- Progressive Distillation Decoder -------------------- #
        pred, pred_z_list, pred_v_list = self.Decoder(*fused_x_e)

        if self.use_ib_loss:
            return pred, pred_z_list, pred_v_list
        else:
            if self.deep_supervision:
                return pred_z_list
            else:
                return pred


if __name__ == '__main__':
    model = MISTNet().cuda()
    inputs = torch.randn((1, 3, 5, 384, 384)).cuda()  # Params = 0.85M FLOPs = 19.31G
    flops, params = profile(model, (inputs,))
    print('Params = ' + str(round(params / 1000 ** 2, 2)) + 'M')
    print('FLOPs = ' + str(round(flops / 1000 ** 3, 2)) + 'G')
    flops, params = clever_format([flops, params], '%.6f')
    print('Params = ' + params)
    print('FLOPs = ' + flops)

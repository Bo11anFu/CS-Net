import torch.nn.functional as F
from torch.nn.init import trunc_normal_
from natten.functional import NATTEN2DQKRPBFunction, NATTEN2DAVFunction
from csnet.models.multiframe.MISTNet.base import *
from thop import profile, clever_format


class ShiftedNeighborhoodCompensationBlock(nn.Module):
    def __init__(self, block, in_channels, shift_size, neighborhood_sizes, dilation_rates=None):
        super(ShiftedNeighborhoodCompensationBlock, self).__init__()
        if shift_size > 0:
            self.shifts = [(delta_h, delta_w) for delta_h in [-shift_size, 0, shift_size]
                           for delta_w in [-shift_size, 0, shift_size] if (delta_h, delta_w) != (0, 0)]
            self.num_groups = len(self.shifts)
            self.conv_smooth = nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=self.num_groups)
            self.conv_reduce = nn.Conv2d(in_channels * 2, in_channels, 1)
        self.shift_size = shift_size

        for neighborhood_size in neighborhood_sizes:
            assert neighborhood_size > 1 and neighborhood_size % 2 == 1, \
                f'Neighborhood size must be an odd number greater than 1, got {neighborhood_size}.'
            assert neighborhood_size in [3, 5, 7, 9, 11, 13], \
                f'CUDA kernel only supports neighborhood sizes 3, 5, 7, 9, 11, and 13, got {neighborhood_size}.'
        self.neighborhood_sizes = neighborhood_sizes
        self.num_scales = len(neighborhood_sizes)
        if dilation_rates is None:
            dilation_rates = [1] * self.num_scales
        else:
            assert len(dilation_rates) == self.num_scales, \
                f'Number of dilation rates must be equal to number of neighborhood sizes.'
            for dilation_rate in dilation_rates:
                assert dilation_rate >= 1, f'Dilation rate must be greater than or equal to 1, got {dilation_rate}.'
        self.dilation_rates = dilation_rates

        self.conv_v_list = nn.ModuleList()
        for i, neighborhood_size in enumerate(neighborhood_sizes):
            self.conv_v_list.append(nn.Conv2d(in_channels, in_channels, 1))
            rpb = nn.Parameter(torch.zeros(1, 2 * neighborhood_size - 1, 2 * neighborhood_size - 1))
            trunc_normal_(rpb, mean=0., std=.02, a=-2., b=2.)
            self.register_parameter(f'rpb{i}', rpb)
        self.conv_fusion = make_layer(block, in_channels * (self.num_scales + 1), in_channels)

    def forward(self, past_frame_feat, curr_frame_feat):
        b, c, h, w = curr_frame_feat.shape
        compensated_feats = [past_frame_feat]

        # Grouped Shift
        if self.shift_size > 0:
            p = self.shift_size
            padded_past_frame_feat = F.pad(past_frame_feat, [p] * 4)  # [b, c, h+2p, w+2p]
            groups = torch.chunk(padded_past_frame_feat, self.num_groups, dim=1)
            shifted_groups = [torch.roll(groups[i], shifts=self.shifts[i], dims=(2, 3)) for i in range(self.num_groups)]
            shifted_past_frame_feat = torch.cat(shifted_groups, dim=1)[..., p:-p, p:-p]  # [b, c, h, w]
            shifted_past_frame_feat = self.conv_smooth(shifted_past_frame_feat)  # [b, c, h, w]
            past_frame_feat = torch.cat([past_frame_feat, shifted_past_frame_feat], dim=1)  # [b, 2c, h, w]
            past_frame_feat = self.conv_reduce(past_frame_feat)  # [b, c, h, w]

        # Multi-neighborhood Matching
        Q = curr_frame_feat.unsqueeze(1).permute(0, 1, 3, 4, 2)  # [b, 1, h, w, c]
        K = past_frame_feat.unsqueeze(1).permute(0, 1, 3, 4, 2)  # [b, 1, h, w, c]
        for i in range(self.num_scales):
            V = self.conv_v_list[i](past_frame_feat).unsqueeze(1).permute(0, 1, 3, 4, 2)  # [b, 1, h, w, c]
            corr_volume = NATTEN2DQKRPBFunction.apply(Q, K, getattr(self, 'rpb{}'.format(i)),
                                                      self.neighborhood_sizes[i], self.dilation_rates[i])
            normalized_corr_volume = corr_volume.softmax(dim=-1)  # [b, 1, h, w, n*n]
            compensated_feat = NATTEN2DAVFunction.apply(normalized_corr_volume, V, self.neighborhood_sizes[i],
                                                        self.dilation_rates[i])  # [b, 1, h, w, c]
            compensated_feat = compensated_feat.permute(0, 1, 4, 2, 3).contiguous().view(b, c, h, w)  # [b, c, h, w]
            compensated_feats.append(compensated_feat)
        compensated_feats = torch.cat(compensated_feats, dim=1)  # [b, (s+1)*c, h, w]

        return self.conv_fusion(compensated_feats)  # [b, c, h, w]


class ImplicitMotionCompensation(nn.Module):
    def __init__(self, block, nb_filter, num_levels, shift_sizes, neighborhood_sizes):
        super(ImplicitMotionCompensation, self).__init__()
        self.num_levels = num_levels
        self.SNCBs = nn.ModuleList([
            ShiftedNeighborhoodCompensationBlock(block, nb_filter[j], shift_sizes[j], neighborhood_sizes)
            for j in range(num_levels)
        ])

    def forward(self, past_frame_feats, curr_frame_feats):
        return [self.SNCBs[j](past_frame_feats[j], curr_frame_feats[j]) for j in range(self.num_levels)]


class MSAM(nn.Module):
    def __init__(self, block, nb_filter, num_frames, num_levels):
        super(MSAM, self).__init__()
        self.num_levels = num_levels
        self.aggregators = nn.ModuleList([
            make_layer(block, nb_filter[j] * num_frames, nb_filter[j])
            for j in range(num_levels)
        ])

    def forward(self, compensated_feats):
        return [self.aggregators[j](compensated_feats[j]) for j in range(self.num_levels)]


class ModulationFilteringBottleneck(nn.Module):
    def __init__(self, block, in_channels, out_channels, modulator):
        super(ModulationFilteringBottleneck, self).__init__()
        self.modulator = modulator(in_channels)
        self.filter = make_layer(block, in_channels, out_channels)

    def forward(self, v):
        return self.filter(self.modulator(v))


class SelfDistillation(nn.Module):
    def __init__(self, block, in_channels, out_channels, num_classes, modulator):
        super(SelfDistillation, self).__init__()
        self.MFB = ModulationFilteringBottleneck(block, in_channels, out_channels, modulator)
        self.head_v = nn.Conv2d(in_channels, num_classes, 1)
        self.head_z = nn.Conv2d(out_channels, num_classes, 1)

    def forward(self, v):
        pred_v = self.head_v(v)  # p(y|v)
        z = self.MFB(v)  # Variational Information Bottleneck: v -> z
        pred_z = self.head_z(z)  # p(y|z)

        return z, pred_z, pred_v


class ProgressiveDistillationDecoder(nn.Module):
    def __init__(self, num_classes, block, nb_filter, modulator):
        super(ProgressiveDistillationDecoder, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.decoder_3 = SelfDistillation(block, nb_filter[2] + nb_filter[3], nb_filter[2], num_classes, modulator)
        self.decoder_2 = SelfDistillation(block, nb_filter[1] + nb_filter[2], nb_filter[1], num_classes, modulator)
        self.decoder_1 = SelfDistillation(block, nb_filter[0] + nb_filter[1], nb_filter[0], num_classes, modulator)

    def forward(self, observations):
        z_4 = observations[3]  # [48, 48]
        z_3, pred_z_3, pred_v_3 = self.decoder_3(torch.cat([observations[2], self.up(observations[3])], 1))  # [96, 96]
        pred_z_3, pred_v_3 = self.up_4(pred_z_3), self.up_4(pred_v_3)  # [384, 384]
        z_2, pred_z_2, pred_v_2 = self.decoder_2(torch.cat([observations[1], self.up(z_3)], 1))  # [192, 192]
        pred_z_2, pred_v_2 = self.up(pred_z_2), self.up(pred_v_2)  # [384, 384]
        z_1, pred_z_1, pred_v_1 = self.decoder_1(torch.cat([observations[0], self.up(z_2)], 1))  # [384, 384]

        return pred_z_1, z_4, z_3, z_2, z_1


class MISTNet_vis_feat(nn.Module):
    def __init__(self, num_frames=5, num_classes=1, in_channels=3, block=ResBlock, num_blocks=[2, 2, 2],
                 nb_filter=[8, 16, 32, 64], shift_sizes=[3, 3, 3, 3], neighborhood_sizes=[3, 5, 7],
                 modulator=CBAM, use_sufficiency_loss=True, deep_supervision=False):
        super(MISTNet_vis_feat, self).__init__()
        self.num_levels = len(nb_filter)
        self.num_frames = num_frames
        self.use_sufficiency_loss = use_sufficiency_loss
        self.deep_supervision = deep_supervision

        self.encoder = ResNet(in_channels, block, num_blocks, nb_filter)
        self.IMC = ImplicitMotionCompensation(block, nb_filter, self.num_levels, shift_sizes, neighborhood_sizes)
        self.MFA = MSAM(block, nb_filter, num_frames, self.num_levels)
        self.decoder = ProgressiveDistillationDecoder(num_classes, block, nb_filter, modulator)

    def forward(self, x):
        curr_frame_feats = self.encoder(x[:, :, -1, :, :])
        all_compensated_feats = [[] for _ in range(self.num_levels)]

        # Implicit Motion Compensation
        for i in range(self.num_frames):
            past_frame_feats = curr_frame_feats if i == self.num_frames - 1 else self.encoder(x[:, :, i, :, :])
            compensated_feats = self.IMC(past_frame_feats, curr_frame_feats)
            for j in range(self.num_levels):
                all_compensated_feats[j].append(compensated_feats[j])

        # Multi-frame Aggregation
        observations = self.MFA([torch.cat(feats, dim=1) for feats in all_compensated_feats])

        # Progressive Distillation Decoder
        pred, z_4, z_3, z_2, z_1 = self.decoder(observations)

        return pred, z_4, z_3, z_2, z_1


if __name__ == '__main__':
    model = MISTNet_vis_feat().cuda()
    inputs = torch.randn((1, 3, 5, 384, 384)).cuda()  # Params = 0.85M FLOPs = 19.31G
    flops, params = profile(model, (inputs,))
    print('Params = ' + str(round(params / 1000 ** 2, 2)) + 'M')
    print('FLOPs = ' + str(round(flops / 1000 ** 3, 2)) + 'G')
    flops, params = clever_format([flops, params], '%.6f')
    print('Params = ' + params)
    print('FLOPs = ' + flops)

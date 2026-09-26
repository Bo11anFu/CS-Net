import torch
import torch.nn as nn
from csnet.models.multiframe.PSTFNet.CIM import CIM
from csnet.models.multiframe.PSTFNet.HDC import HDC_module
from csnet.models.multiframe.PSTFNet.block import Res_block, Res_block_CBAM
from thop import profile, clever_format


class PSTFNet(nn.Module):
    # 4 stage
    def __init__(self, num_classes=1, input_channels=3, block=Res_block_CBAM, num_blocks=[2, 2, 2, 2],
                 nb_filter=[8, 16, 32, 64, 128], deep_supervision=False):
        super(PSTFNet, self).__init__()

        self.deep_supervision = deep_supervision
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = self._make_layer(block, input_channels, nb_filter[0], num_blocks[0])
        self.conv1_0 = self._make_layer(block, nb_filter[0], nb_filter[1], num_blocks[0])
        self.conv2_0 = self._make_layer(block, nb_filter[1], nb_filter[2], num_blocks[1])
        self.conv3_0 = self._make_layer(block, nb_filter[2], nb_filter[3], num_blocks[2])

        self.conv2_1 = self._make_layer(Res_block, nb_filter[2] + nb_filter[3], nb_filter[2], num_blocks[2])
        self.conv1_1 = self._make_layer(Res_block, nb_filter[1] + nb_filter[2], nb_filter[1], num_blocks[1])
        self.conv0_1 = self._make_layer(Res_block, nb_filter[0] + nb_filter[1], nb_filter[0], num_blocks[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

        self.CIM = CIM(120)

        self.hdc0 = HDC_module(8, 8)
        self.hdc1 = HDC_module(16, 16)
        self.hdc2 = HDC_module(32, 32)
        self.hdc3 = HDC_module(64, 64)

        if self.deep_supervision:
            self.final2_1 = nn.Conv2d(nb_filter[2], num_classes, kernel_size=1)
            self.final1_1 = nn.Conv2d(nb_filter[1], num_classes, kernel_size=1)
            self.up_to_input_size = nn.Upsample(size=(512, 512), mode='bilinear', align_corners=True)

    def _make_layer(self, block, input_channels, output_channels, num_blocks=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(num_blocks - 1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward(self, input):

        frameNum = input.shape[2]
        features_0 = []
        features_1 = []
        features_2 = []
        features_3 = []
        for i in range(frameNum):
            x = input[:, :, i, :, :]

            x0_0 = self.conv0_0(x)
            x1_0 = self.conv1_0(self.pool(x0_0))
            x2_0 = self.conv2_0(self.pool(x1_0))
            x3_0 = self.conv3_0(self.pool(x2_0))
            features_0.append(x0_0)
            features_1.append(x1_0)
            features_2.append(x2_0)
            features_3.append(x3_0)
        features_0 = torch.stack(features_0, dim=2)
        features_1 = torch.stack(features_1, dim=2)
        features_2 = torch.stack(features_2, dim=2)
        features_3 = torch.stack(features_3, dim=2)

        x0_0_residual = self.hdc0(features_0)
        x1_0_residual = self.hdc1(features_1)
        x2_0_residual = self.hdc2(features_2)
        x3_0_residual = self.hdc3(features_3)

        attention = self.CIM(tuple([x0_0_residual, x1_0_residual, x2_0_residual, x3_0_residual]))

        x2_1 = self.conv2_1(torch.cat([attention[2], self.up(attention[3])], 1))
        x1_1 = self.conv1_1(torch.cat([attention[1], self.up(x2_1)], 1))
        x0_1 = self.conv0_1(torch.cat([attention[0], self.up(x1_1)], 1))

        output = self.final(x0_1)
        return output


if __name__ == '__main__':
    model = PSTFNet().cuda()
    inputs = torch.randn((1, 3, 5, 384, 384)).cuda()  # Params = 1.65M FLOPs = 18.19G
    flops, params = profile(model, (inputs,))
    print('Params = ' + str(round(params / 1000 ** 2, 2)) + 'M')
    print('FLOPs = ' + str(round(flops / 1000 ** 3, 2)) + 'G')
    flops, params = clever_format([flops, params], '%.6f')
    print('Params = ' + params)
    print('FLOPs = ' + flops)

import torch
import torch.nn as nn
from csnet.models.multiframe.DTUM.layers_vis import DTUM
from thop import profile, clever_format


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


class ResUNet(nn.Module):
    def __init__(self, num_classes, input_channels, block, num_blocks, nb_filter):
        super(ResUNet, self).__init__()

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = self._make_layer(block, input_channels, nb_filter[0])
        self.conv1_0 = self._make_layer(block, nb_filter[0], nb_filter[1], num_blocks[0])
        self.conv2_0 = self._make_layer(block, nb_filter[1], nb_filter[2], num_blocks[1])
        self.conv3_0 = self._make_layer(block, nb_filter[2], nb_filter[3], num_blocks[2])
        # self.conv4_0 = self._make_layer(block, nb_filter[3],   nb_filter[4], num_blocks[3])

        # self.conv3_1 = self._make_layer(block, nb_filter[3] + nb_filter[4], nb_filter[3])
        # self.conv2_2 = self._make_layer(block, nb_filter[2] + nb_filter[3], nb_filter[2])
        # self.conv1_3 = self._make_layer(block, nb_filter[1] + nb_filter[2], nb_filter[1])
        # self.conv0_4 = self._make_layer(block, nb_filter[0] + nb_filter[1], nb_filter[0])

        self.conv2_1 = self._make_layer(block, nb_filter[2] + nb_filter[3], nb_filter[2])
        self.conv1_2 = self._make_layer(block, nb_filter[1] + nb_filter[2], nb_filter[1])
        self.conv0_3 = self._make_layer(block, nb_filter[0] + nb_filter[1], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def _make_layer(self, block, input_channels, output_channels, num_blocks=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(num_blocks - 1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        # x4_0 = self.conv4_0(self.pool(x3_0))

        # x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        # x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        # x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        # x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, self.up(x1_2)], 1))

        # output = self.final(x0_4)
        output = self.final(x0_3)
        return output


class ResUNet_DTUM_vis_feat(nn.Module):
    def __init__(self, num_classes=1, input_channels=3, block=Res_block, num_blocks=[2, 2, 2, 2],
                 nb_filter=[8, 16, 32, 64, 128]):
        super(ResUNet_DTUM_vis_feat, self).__init__()

        self.UNet = ResUNet(32, input_channels, block, num_blocks, nb_filter)
        self.DTUM = DTUM(32, num_classes, num_frames=5)

    def forward(self, X_In):
        FrameNum = X_In.shape[2]  # 5

        # key frame
        Features = X_In[:, :, -1, :, :]  # (B, 3, H, W)
        Features = self.UNet(Features)  # (B, 32, H, W)
        Features = torch.unsqueeze(Features, 2)  # (B, 32, 1, H, W)

        # sup frame
        for i_fra in range(FrameNum - 1):  # 0,1,2,3
            x_t = X_In[:, :, -2 - i_fra, :, :]
            x_t = self.UNet(x_t)
            x_t = torch.unsqueeze(x_t, 2)
            Features = torch.cat([x_t, Features], 2)  # t-4, t-3, t-2, t-1, t

        X_Out, z_4, z_3, z_2, z_1 = self.DTUM(Features)

        return X_Out, z_4, z_3, z_2, z_1


if __name__ == '__main__':
    # model = ResUNet_DTUM(input_channels=1).cuda()
    # inputs = torch.randn((4, 1, 5, 256, 256)).cuda()  # Params = 0.3M FLOPs = 41.19G

    model = ResUNet_DTUM_vis_feat(input_channels=3).cuda()
    inputs = torch.randn((1, 3, 5, 384, 384)).cuda()  # Params = 0.3M FLOPs = 23.29G

    flops, params = profile(model, (inputs,))
    print('Params = ' + str(round(params / 1000 ** 2, 2)) + 'M')
    print('FLOPs = ' + str(round(flops / 1000 ** 3, 2)) + 'G')
    flops, params = clever_format([flops, params], '%.6f')
    print('Params = ' + params)
    print('FLOPs = ' + flops)

from importlib import util
from pathlib import Path

import torch
import torch.nn as nn


from csnet.models.singleframe.DNANet.model_DNANet import (
    ChannelAttention,
    DNANet,
    Res_CBAM_block,
    SpatialAttention,
)


def _load_dhif_layer():
    basic_path = Path(__file__).resolve().parent / "model" / "basic.py"
    spec = util.spec_from_file_location("_dhif_basic", basic_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load DHiF layer from {basic_path}")

    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DHiF


DHiF = _load_dhif_layer()


class Res_CBAM_block_DHiF(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = DHiF(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = None

        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ca(out) * out
        out = self.sa(out) * out
        out += residual
        out = self.relu(out)
        return out


class DNANet_DHiF(DNANet):
    def __init__(
        self,
        num_classes=1,
        input_channels=3,
        block=None,
        num_blocks=[2, 2, 2, 2],
        nb_filter=[16, 32, 64, 128, 256],
        deep_supervision=False,
    ):
        super().__init__(
            num_classes=num_classes,
            input_channels=input_channels,
            block=block or Res_CBAM_block,
            num_blocks=num_blocks,
            nb_filter=nb_filter,
            deep_supervision=deep_supervision,
        )

        decoder_block = Res_CBAM_block_DHiF

        self.conv0_1 = self._make_layer(decoder_block, nb_filter[0] + nb_filter[1], nb_filter[0])
        self.conv1_1 = self._make_layer(
            decoder_block,
            nb_filter[1] + nb_filter[2] + nb_filter[0],
            nb_filter[1],
            num_blocks[0],
        )
        self.conv2_1 = self._make_layer(
            decoder_block,
            nb_filter[2] + nb_filter[3] + nb_filter[1],
            nb_filter[2],
            num_blocks[1],
        )
        self.conv3_1 = self._make_layer(
            decoder_block,
            nb_filter[3] + nb_filter[4] + nb_filter[2],
            nb_filter[3],
            num_blocks[2],
        )

        self.conv0_2 = self._make_layer(decoder_block, nb_filter[0] * 2 + nb_filter[1], nb_filter[0])
        self.conv1_2 = self._make_layer(
            decoder_block,
            nb_filter[1] * 2 + nb_filter[2] + nb_filter[0],
            nb_filter[1],
            num_blocks[0],
        )
        self.conv2_2 = self._make_layer(
            decoder_block,
            nb_filter[2] * 2 + nb_filter[3] + nb_filter[1],
            nb_filter[2],
            num_blocks[1],
        )

        self.conv0_3 = self._make_layer(decoder_block, nb_filter[0] * 3 + nb_filter[1], nb_filter[0])
        self.conv1_3 = self._make_layer(
            decoder_block,
            nb_filter[1] * 3 + nb_filter[2] + nb_filter[0],
            nb_filter[1],
            num_blocks[0],
        )

        self.conv0_4 = self._make_layer(decoder_block, nb_filter[0] * 4 + nb_filter[1], nb_filter[0])
        self.conv0_4_final = self._make_layer(decoder_block, nb_filter[0] * 5, nb_filter[0])


__all__ = ["DNANet_DHiF", "Res_CBAM_block_DHiF"]


if __name__ == "__main__":
    from thop import clever_format, profile

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DNANet_DHiF(input_channels=3).to(device)
    inputs = torch.randn((1, 3, 384, 384), device=device)

    flops, params = profile(model, (inputs,))
    print("Params = " + str(round(params / 1000 ** 2, 2)) + "M")
    print("FLOPs = " + str(round(flops / 1000 ** 3, 2)) + "G")
    flops, params = clever_format([flops, params], "%.6f")
    print("Params = " + params)
    print("FLOPs = " + flops)

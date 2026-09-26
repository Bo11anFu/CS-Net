from importlib import util
from pathlib import Path

import torch
import torch.nn as nn


from csnet.models.singleframe.MSHNet.model_MSHNet import (
    ChannelAttention,
    MSHNet,
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


class ResNet_DHiF(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_pconv=False):
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


class MSHNet_DHiF(MSHNet):
    def __init__(self, input_channels=3, block=None, use_pconv=False):
        super().__init__(input_channels=input_channels, use_pconv=False)
        param_channels = [16, 32, 64, 128, 256]
        param_blocks = [2, 2, 2, 2]

        self.encoder_0 = self._make_layer(param_channels[0], param_channels[0], ResNet_DHiF)
        self.encoder_1 = self._make_layer(
            param_channels[0],
            param_channels[1],
            ResNet_DHiF,
            param_blocks[0],
        )


__all__ = ["MSHNet_DHiF", "ResNet_DHiF"]


if __name__ == "__main__":
    from thop import clever_format, profile

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MSHNet_DHiF(use_pconv=False).to(device)
    inputs = torch.randn((1, 3, 384, 384), device=device)

    flops, params = profile(model, (inputs,))
    print("Params = " + str(round(params / 1000 ** 2, 2)) + "M")
    print("FLOPs = " + str(round(flops / 1000 ** 3, 2)) + "G")
    flops, params = clever_format([flops, params], "%.6f")
    print("Params = " + params)
    print("FLOPs = " + flops)

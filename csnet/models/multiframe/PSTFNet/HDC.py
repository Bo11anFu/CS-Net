import torch
import torch.nn as nn
from torch.nn import functional as F


class Conv_1x1x1(nn.Module):
    def __init__(self, in_dim, out_dim, activation):
        super(Conv_1x1x1, self).__init__()
        self.conv1 = nn.Conv3d(in_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.norm = nn.BatchNorm3d(out_dim)
        self.act = activation

    def forward(self, x):
        x = self.act(self.norm(self.conv1(x)))
        return x


class Conv_4x3x3(nn.Module):
    def __init__(self, in_dim, out_dim, activation):
        super(Conv_4x3x3, self).__init__()
        self.conv1 = nn.Conv3d(in_dim, out_dim, kernel_size=(4, 3, 3), stride=1, padding=(0, 1, 1), bias=True)
        self.norm = nn.BatchNorm3d(out_dim)
        self.act = activation

    def forward(self, x):
        x = self.act(self.norm(self.conv1(x)))
        return x


class Conv_4x1x1(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Conv_4x1x1, self).__init__()
        self.conv1 = nn.Conv3d(in_dim, out_dim, kernel_size=(4, 1, 1), stride=1, padding=(0, 0, 0), bias=True)
        self.norm = nn.BatchNorm3d(out_dim)

    def forward(self, x):
        x = self.norm(self.conv1(x))
        return x


class Conv_3x1x1(nn.Module):
    def __init__(self, in_dim, out_dim, activation):
        super(Conv_3x1x1, self).__init__()
        self.conv1 = nn.Conv3d(in_dim, out_dim, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0), bias=True)
        self.norm = nn.BatchNorm3d(out_dim)
        self.act = activation

    def forward(self, x):
        x = self.act(self.norm(self.conv1(x)))
        return x


class Conv_1x3x3(nn.Module):
    def __init__(self, in_dim, out_dim, activation):
        super(Conv_1x3x3, self).__init__()
        self.conv1 = nn.Conv3d(in_dim, out_dim, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), bias=True)
        self.norm = nn.BatchNorm3d(out_dim)
        self.act = activation

    def forward(self, x):
        x = self.act(self.norm(self.conv1(x)))
        return x


class Conv_5x1x1(nn.Module):
    def __init__(self, in_dim, out_dim, activation):
        super(Conv_5x1x1, self).__init__()
        self.conv1 = nn.Conv3d(in_dim, out_dim, kernel_size=(5, 1, 1), stride=1, padding=(2, 0, 0), bias=True)
        self.norm = nn.BatchNorm3d(out_dim)
        self.act = activation

    def forward(self, x):
        x = self.act(self.norm(self.conv1(x)))
        return x


class Conv_3x3x3(nn.Module):
    def __init__(self, in_dim, out_dim, activation):
        super(Conv_3x3x3, self).__init__()
        self.conv1 = nn.Conv3d(in_dim, out_dim, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1), bias=True)
        self.norm = nn.BatchNorm3d(out_dim)
        self.act = activation

    def forward(self, x):
        x = self.act(self.norm(self.conv1(x)))
        return x


class Conv_5x3x3(nn.Module):
    def __init__(self, in_dim, out_dim, activation):
        super(Conv_5x3x3, self).__init__()
        self.conv1 = nn.Conv3d(in_dim, out_dim, kernel_size=(5, 3, 3), stride=1, padding=(2, 1, 1), bias=True)
        self.norm = nn.BatchNorm3d(out_dim)
        self.act = activation

    def forward(self, x):
        x = self.act(self.norm(self.conv1(x)))
        return x


class Conv_1x1(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Conv_1x1, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size=(1, 1), stride=1, padding=(0, 0), bias=True)
        self.norm = nn.BatchNorm2d(out_dim)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.act(self.norm(self.conv1(x)))
        return x


class Conv_3x3(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Conv_3x3, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size=(3, 3), stride=1, padding=(1, 1), bias=True)
        self.norm = nn.BatchNorm2d(out_dim)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.act(self.norm(self.conv1(x)))
        return x


class Conv_5x5(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Conv_5x5, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size=(5, 5), stride=1, padding=(2, 2), bias=True)
        self.norm = nn.BatchNorm2d(out_dim)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.act(self.norm(self.conv1(x)))
        return x


class Conv_7x7(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Conv_7x7, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size=(7, 7), stride=1, padding=(3, 3), bias=True)
        self.norm = nn.BatchNorm2d(out_dim)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.act(self.norm(self.conv1(x)))
        return x


class Conv_9x9(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Conv_9x9, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size=(9, 9), stride=1, padding=(4, 4), bias=True)
        self.norm = nn.BatchNorm2d(out_dim)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.act(self.norm(self.conv1(x)))
        return x


# class HDC_module(nn.Module):
#     def __init__(self, in_dim, out_dim, activation=nn.ReLU(inplace=True)):
#         super(HDC_module, self).__init__()
#         self.in_dim = in_dim
#         self.out_dim = out_dim
#         self.inter_dim = in_dim//4
#
#         self.conv_1x1x1_1 = Conv_1x1x1(self.in_dim, self.inter_dim, activation)
#         self.conv_1x1x1_2 = Conv_1x1x1(self.inter_dim, self.out_dim, activation)
#
#         self.conv_3x3x3_1 = Conv_3x3x3(self.inter_dim, self.inter_dim, activation)
#         self.conv_3x3x3_2 = Conv_3x3x3(self.inter_dim, self.inter_dim, activation)
#
#
#
#     def forward(self, x):
#         x = self.conv_1x1x1_1(x)
#         x = self.conv_3x3x3_1(x)
#         x = self.conv_3x3x3_2(x)
#         x = self.conv_1x1x1_2(x)
#
#         return x[:, :, -1, :, :]
#
# class HDC_module(nn.Module):
#     def __init__(self, in_dim, out_dim, activation=nn.ReLU(inplace=True)):
#         super(HDC_module, self).__init__()
#         self.in_dim = in_dim
#         self.out_dim = out_dim
#         self.inter_dim = in_dim//4
#
#         self.head = Conv_1x1x1(self.in_dim, self.inter_dim, activation)
#         self.tail = Conv_1x1x1(self.inter_dim * 3, self.out_dim, activation)
#
#         self.conv_1x1x1_1 = Conv_1x1x1(self.inter_dim, self.inter_dim, activation)
#         self.conv_1x1x1_2 = Conv_1x1x1(self.inter_dim, self.inter_dim, activation)
#
#         self.conv_1x3x3_3 = Conv_1x1x1(self.inter_dim, self.inter_dim, activation)
#         self.conv_3x1x1 = Conv_3x1x1(self.inter_dim, self.inter_dim, activation)
#
#         self.conv_1x3x3_5 = Conv_1x3x3(self.inter_dim, self.inter_dim, activation)
#         self.conv_5x1x1 = Conv_5x1x1(self.inter_dim, self.inter_dim, activation)
#
#     def forward(self, x):
#         x = self.head(x)
#         x1 = self.conv_1x1x1_2(self.conv_1x1x1_1(x))
#         x3 = self.conv_3x1x1(self.conv_1x3x3_3(x))
#         x5 = self.conv_5x1x1(self.conv_1x3x3_5(x))
#         x = self.tail(torch.cat([x1, x3, x5], dim=1))
#
#         return x[:, :, -1, :, :]


# class HDC_module(nn.Module):
#     def __init__(self, in_dim, out_dim, activation=nn.ReLU(inplace=True)):
#         super(HDC_module, self).__init__()
#         self.in_dim = in_dim
#         self.out_dim = out_dim
#         self.inter_dim = in_dim//4
#
#         self.head = Conv_3x3x3(self.in_dim, self.inter_dim, activation)
#
#         self.conv_1x3x3_1 = Conv_1x3x3(self.inter_dim, self.inter_dim, activation)
#         self.conv_1x1x1_2 = Conv_1x1x1(self.inter_dim, self.inter_dim, activation)
#         self.conv_1x3x3_3 = Conv_1x3x3(self.inter_dim, self.inter_dim, activation)
#         self.conv_1x1x1_4 = Conv_1x1x1(self.inter_dim, self.inter_dim, activation)
#
#         self.conv_1x3x3_3_1 = Conv_1x3x3(self.inter_dim, self.inter_dim, activation)
#         self.conv_3x1x1_1 = Conv_3x1x1(self.inter_dim, self.inter_dim, activation)
#         self.conv_1x3x3_3_2 = Conv_1x3x3(self.inter_dim, self.inter_dim, activation)
#         self.conv_3x1x1_2 = Conv_3x1x1(self.inter_dim, self.inter_dim, activation)
#
#         self.conv_1x3x3_5_1 = Conv_1x3x3(self.inter_dim, self.inter_dim, activation)
#         self.conv_5x1x1_1 = Conv_5x1x1(self.inter_dim, self.inter_dim, activation)
#         self.conv_1x3x3_5_2 = Conv_1x3x3(self.inter_dim, self.inter_dim, activation)
#         self.conv_5x1x1_2 = Conv_5x1x1(self.inter_dim, self.inter_dim, activation)
#
#         self.tail = Conv_3x3x3(self.inter_dim * 3, self.inter_dim, activation)
#         self.last = Conv_4x1x1(self.inter_dim, self.out_dim)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         x = self.head(x)
#         x1 = self.conv_1x1x1_4(self.conv_1x3x3_3(self.conv_1x1x1_2(self.conv_1x3x3_1(x))))
#         x3 = self.conv_3x1x1_2(self.conv_1x3x3_3_2(self.conv_3x1x1_1(self.conv_1x3x3_3_1(x))))
#         x5 = self.conv_5x1x1_2(self.conv_1x3x3_5_2(self.conv_5x1x1_1(self.conv_1x3x3_5_1(x))))
#         x = self.tail(torch.cat([x1, x3, x5], dim=1))
#         x = self.last(x)
#         x = torch.squeeze(x, dim=2)
#
#         return self.sigmoid(x)

# class HDC_module(nn.Module):
#     def __init__(self, in_dim, out_dim, activation=nn.ReLU(inplace=True)):
#         super(HDC_module, self).__init__()
#         self.in_dim = in_dim
#         self.out_dim = out_dim
#         self.inter_dim = in_dim//4
#
#         self.head = Conv_3x3x3(self.in_dim, self.inter_dim, activation)
#
#         self.conv_1x3x3_1 = Conv_1x3x3(self.inter_dim, self.inter_dim, activation)
#         self.conv_1x3x3_2 = Conv_1x3x3(self.inter_dim, self.inter_dim, activation)
#
#         self.conv_3x3x3_1 = Conv_3x3x3(self.inter_dim, self.inter_dim, activation)
#         self.conv_3x3x3_2 = Conv_3x3x3(self.inter_dim, self.inter_dim, activation)
#
#         self.conv_5x3x3_1 = Conv_5x3x3(self.inter_dim, self.inter_dim, activation)
#         self.conv_5x3x3_2 = Conv_5x3x3(self.inter_dim, self.inter_dim, activation)
#
#         self.tail = Conv_3x3x3(self.inter_dim * 3, self.inter_dim, activation)
#         self.last = Conv_4x1x1(self.inter_dim, self.out_dim)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         x = self.head(x)
#         x1 = self.conv_1x3x3_2(self.conv_1x3x3_1(x))
#         x3 = self.conv_3x3x3_2(self.conv_3x3x3_1(x))
#         x5 = self.conv_5x3x3_2(self.conv_5x3x3_1(x))
#         x = self.tail(torch.cat([x1, x3, x5], dim=1))
#         x = self.last(x)
#
#         x = torch.squeeze(x, dim=2)
#
#         return self.sigmoid(x)


# class HDC_module(nn.Module):
#     def __init__(self, in_dim, out_dim, activation=nn.ReLU(inplace=True)):
#         super(HDC_module, self).__init__()
#         self.in_dim = in_dim
#         self.out_dim = out_dim
#         self.inter_dim = in_dim//4
#
#         self.head = Conv_3x3x3(self.in_dim, self.inter_dim, activation)
#
#         self.conv_1x3x3_1 = Conv_1x3x3(self.inter_dim, self.inter_dim, activation)
#         self.conv_1x1x1_2 = Conv_1x1x1(self.inter_dim, self.inter_dim, activation)
#         self.conv_1x3x3_3 = Conv_1x3x3(self.inter_dim, self.inter_dim, activation)
#         self.conv_1x1x1_4 = Conv_1x1x1(self.inter_dim, self.inter_dim, activation)
#
#         self.conv_1x3x3_3_1 = Conv_1x3x3(self.inter_dim, self.inter_dim, activation)
#         self.conv_3x1x1_1 = Conv_3x1x1(self.inter_dim, self.inter_dim, activation)
#         self.conv_1x3x3_3_2 = Conv_1x3x3(self.inter_dim, self.inter_dim, activation)
#         self.conv_3x1x1_2 = Conv_3x1x1(self.inter_dim, self.inter_dim, activation)
#
#         self.conv_1x3x3_5_1 = Conv_1x3x3(self.inter_dim, self.inter_dim, activation)
#         self.conv_5x1x1_1 = Conv_5x1x1(self.inter_dim, self.inter_dim, activation)
#         self.conv_1x3x3_5_2 = Conv_1x3x3(self.inter_dim, self.inter_dim, activation)
#         self.conv_5x1x1_2 = Conv_5x1x1(self.inter_dim, self.inter_dim, activation)
#
#         self.tail = Conv_3x3x3(self.inter_dim * 3, self.inter_dim, activation)
#         self.last = Conv_4x1x1(self.inter_dim, self.out_dim)
#         self.sigmoid = nn.Sigmoid()
#
#         self.conv1 = Conv_3x3x3(self.in_dim, self.inter_dim, activation)
#         self.conv2 = Conv_3x3x3(self.inter_dim, self.out_dim, activation)
#
#         self.conv3_3 = Conv_1x1(self.inter_dim, self.inter_dim)
#         self.conv2_2 = Conv_1x1(self.inter_dim, self.inter_dim)
#         self.conv1_1 = Conv_1x1(self.inter_dim, self.inter_dim)
#         self.conv0_0 = Conv_1x1(self.inter_dim, self.inter_dim)
#
#         self.conv3_2 = Conv_3x3(self.inter_dim, self.inter_dim)
#         self.conv2_1 = Conv_3x3(self.inter_dim, self.inter_dim)
#         self.conv1_0 = Conv_3x3(self.inter_dim, self.inter_dim)
#
#         self.conv3_1 = Conv_5x5(self.inter_dim, self.inter_dim)
#         self.conv2_0 = Conv_5x5(self.inter_dim, self.inter_dim)
#
#         self.conv3_0 = Conv_7x7(self.inter_dim, self.inter_dim)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         t = x[:, :, -1, :, :]
#         t_1 = x[:, :, -2, :, :]
#         t_2 = x[:, :, -3, :, :]
#         t_3 = x[:, :, -4, :, :]
#
#         t_3 = self.conv3_3(t_3)
#         t_2 = self.conv2_2(t_2) + self.conv3_2(t_2 + t_3)
#         t_1 = self.conv1_1(t_1) + self.conv2_1(t_1 + t_2) + self.conv3_1(t_1 + t_3)
#         t = self.conv0_0(t) + self.conv1_0(t + t_1) + self.conv2_0(t + t_2) + self.conv3_0(t + t_3)
#
#         x = torch.stack([t_3, t_2, t_1, t], dim=2)
#         x = self.conv2(x)
#
#         residual = x[:, :, -1, :, :]
#         x = self.head(x)
#         x1 = self.conv_1x1x1_4(self.conv_1x3x3_3(self.conv_1x1x1_2(self.conv_1x3x3_1(x))))
#         x3 = self.conv_3x1x1_2(self.conv_1x3x3_3_2(self.conv_3x1x1_1(self.conv_1x3x3_3_1(x))))
#         x5 = self.conv_5x1x1_2(self.conv_1x3x3_5_2(self.conv_5x1x1_1(self.conv_1x3x3_5_1(x))))
#         x = self.tail(torch.cat([x1, x3, x5], dim=1))
#         x = self.last(x)
#         x = torch.squeeze(x, dim=2)
#         x = self.sigmoid(x)
#         x = x - 0.5
#         out = residual + residual * x
#
#         return out


class DT_module(nn.Module):
    def __init__(self, in_dim, out_dim, activation=nn.ReLU(inplace=True)):
        super(DT_module, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.inter_dim = in_dim // 4
        self.conv1 = Conv_3x3x3(self.in_dim, self.inter_dim, activation)
        self.conv2 = Conv_3x3x3(self.inter_dim, self.out_dim, activation)

        self.conv4_4 = Conv_1x1(self.inter_dim, self.inter_dim)
        self.conv3_3 = Conv_1x1(self.inter_dim, self.inter_dim)
        self.conv2_2 = Conv_1x1(self.inter_dim, self.inter_dim)
        self.conv1_1 = Conv_1x1(self.inter_dim, self.inter_dim)
        self.conv0_0 = Conv_1x1(self.inter_dim, self.inter_dim)

        self.conv4_3 = Conv_3x3(self.inter_dim, self.inter_dim)
        self.conv3_2 = Conv_3x3(self.inter_dim, self.inter_dim)
        self.conv2_1 = Conv_3x3(self.inter_dim, self.inter_dim)
        self.conv1_0 = Conv_3x3(self.inter_dim, self.inter_dim)

        self.conv4_2 = Conv_5x5(self.inter_dim, self.inter_dim)
        self.conv3_1 = Conv_5x5(self.inter_dim, self.inter_dim)
        self.conv2_0 = Conv_5x5(self.inter_dim, self.inter_dim)

        self.conv4_1 = Conv_7x7(self.inter_dim, self.inter_dim)
        self.conv3_0 = Conv_7x7(self.inter_dim, self.inter_dim)

        self.conv4_0 = Conv_9x9(self.inter_dim, self.inter_dim)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)

        t = x[:, :, -1, :, :]
        t_1 = x[:, :, -2, :, :]
        t_2 = x[:, :, -3, :, :]
        t_3 = x[:, :, -4, :, :]
        t_4 = x[:, :, -5, :, :]

        t_4 = self.conv4_4(t_4)
        t_3 = self.conv3_3(t_3) + self.conv4_3(t_3 + t_4)
        t_2 = self.conv2_2(t_2) + self.conv3_2(t_2 + t_3) + self.conv4_2(t_2 + t_4)
        t_1 = self.conv1_1(t_1) + self.conv2_1(t_1 + t_2) + self.conv3_1(t_1 + t_3) + self.conv4_1(t_1 + t_4)
        t = self.conv0_0(t) + self.conv1_0(t + t_1) + self.conv2_0(t + t_2) + self.conv3_0(t + t_3) + self.conv4_0(
            t + t_4)

        x = torch.stack([t_4, t_3, t_2, t_1, t], dim=2)
        x = self.conv2(x)

        out = self.sigmoid(x[:, :, -1, :, :])

        return out


# class DT_module(nn.Module):
#     def __init__(self, in_dim, out_dim, activation=nn.ReLU(inplace=True)):
#         super(DT_module, self).__init__()
#         self.in_dim = in_dim
#         self.out_dim = out_dim
#         self.inter_dim = in_dim // 4
#         self.conv1 = Conv_3x3x3(self.in_dim, self.inter_dim, activation)
#         self.conv2 = Conv_3x3x3(self.inter_dim, self.out_dim, activation)
#
#         self.conv3_3 = Conv_1x1(self.inter_dim, self.inter_dim)
#         self.conv2_2 = Conv_1x1(self.inter_dim, self.inter_dim)
#         self.conv1_1 = Conv_1x1(self.inter_dim, self.inter_dim)
#         self.conv0_0 = Conv_1x1(self.inter_dim, self.inter_dim)
#
#         self.conv3_2 = Conv_3x3(self.inter_dim, self.inter_dim)
#         self.conv2_1 = Conv_3x3(self.inter_dim, self.inter_dim)
#         self.conv1_0 = Conv_3x3(self.inter_dim, self.inter_dim)
#
#         self.conv3_1 = Conv_5x5(self.inter_dim, self.inter_dim)
#         self.conv2_0 = Conv_5x5(self.inter_dim, self.inter_dim)
#
#         self.conv3_0 = Conv_7x7(self.inter_dim, self.inter_dim)
#
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#
#         x = self.conv1(x)
#
#         t = x[:, :, -1, :, :]
#         t_1 = x[:, :, -2, :, :]
#         t_2 = x[:, :, -3, :, :]
#         t_3 = x[:, :, -4, :, :]
#
#         t_3 = self.conv3_3(t_3)
#         t_2 = self.conv2_2(t_2) + self.conv3_2(t_2 + t_3)
#         t_1 = self.conv1_1(t_1) + self.conv2_1(t_1 + t_2) + self.conv3_1(t_1 + t_3)
#         t = self.conv0_0(t) + self.conv1_0(t + t_1) + self.conv2_0(t + t_2) + self.conv3_0(t + t_3)
#
#         x = torch.stack([t_3, t_2, t_1, t], dim=2)
#         x = self.conv2(x)
#
#         out = self.sigmoid(x[:, :, -1, :, :])
#
#         return out

# class DT_module(nn.Module):
#     def __init__(self, in_dim, out_dim, activation=nn.ReLU(inplace=True)):
#         super(DT_module, self).__init__()
#         self.in_dim = in_dim
#         self.out_dim = out_dim
#         self.inter_dim = in_dim//4
#         self.conv1 = Conv_3x3x3(self.in_dim, self.inter_dim * 4, activation)
#         self.conv2 = Conv_3x3x3(self.inter_dim * 4, self.out_dim, activation)
#
#         self.conv_0 = Conv_3x3x3(self.inter_dim, self.inter_dim, activation)
#         self.conv_1 = Conv_3x3x3(self.inter_dim, self.inter_dim, activation)
#         self.conv_2 = Conv_3x3x3(self.inter_dim, self.inter_dim, activation)
#         self.conv_3 = Conv_3x3x3(self.inter_dim, self.inter_dim, activation)
#
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         # x = self.conv1(x)
#
#         t = x[:, :self.inter_dim, :, :, :]
#         t_1 = x[:, self.inter_dim:2*self.inter_dim, :, :, :]
#         t_2 = x[:, 2*self.inter_dim:3*self.inter_dim, :, :, :]
#         t_3 = x[:, 3*self.inter_dim:, :, :, :]
#
#         t_3 = self.conv_3(t_3)
#         t_2 = self.conv_2(t_2 + t_3)
#         t_1 = self.conv_1(t_1 + t_2)
#         t = self.conv_0(t + t_1)
#
#         x = torch.cat([t_3, t_2, t_1, t], dim=1)
#         x = self.conv2(x)
#
#         out = self.sigmoid(x[:, :, -1, :, :])
#
#         return out

class HDC_module(nn.Module):
    def __init__(self, in_dim, out_dim, activation=nn.ReLU(inplace=True)):
        super(HDC_module, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.inter_dim = in_dim // 4

        self.head = Conv_3x3x3(self.in_dim, self.inter_dim, activation)

        self.conv_1x3x3_1 = Conv_1x3x3(self.inter_dim, self.inter_dim, activation)
        self.conv_1x1x1_2 = Conv_1x1x1(self.inter_dim, self.inter_dim, activation)
        self.conv_1x3x3_3 = Conv_1x3x3(self.inter_dim, self.inter_dim, activation)
        self.conv_1x1x1_4 = Conv_1x1x1(self.inter_dim, self.inter_dim, activation)

        self.conv_1x3x3_3_1 = Conv_1x3x3(self.inter_dim, self.inter_dim, activation)
        self.conv_3x1x1_1 = Conv_3x1x1(self.inter_dim, self.inter_dim, activation)
        self.conv_1x3x3_3_2 = Conv_1x3x3(self.inter_dim, self.inter_dim, activation)
        self.conv_3x1x1_2 = Conv_3x1x1(self.inter_dim, self.inter_dim, activation)

        self.conv_1x3x3_5_1 = Conv_1x3x3(self.inter_dim, self.inter_dim, activation)
        self.conv_5x1x1_1 = Conv_5x1x1(self.inter_dim, self.inter_dim, activation)
        self.conv_1x3x3_5_2 = Conv_1x3x3(self.inter_dim, self.inter_dim, activation)
        self.conv_5x1x1_2 = Conv_5x1x1(self.inter_dim, self.inter_dim, activation)

        self.tail = Conv_3x3x3(self.inter_dim * 3, self.out_dim, activation)
        # self.last = Conv_4x1x1(self.inter_dim, self.out_dim)
        self.sigmoid = nn.Sigmoid()

        self.DT = DT_module(in_dim, out_dim, activation)

    def forward(self, x):
        residual = x[:, :, -1, :, :]

        out1 = self.DT(x)

        x = self.head(x)
        x1 = self.conv_1x1x1_4(self.conv_1x3x3_3(self.conv_1x1x1_2(self.conv_1x3x3_1(x))))
        x3 = self.conv_3x1x1_2(self.conv_1x3x3_3_2(self.conv_3x1x1_1(self.conv_1x3x3_3_1(x))))
        x5 = self.conv_5x1x1_2(self.conv_1x3x3_5_2(self.conv_5x1x1_1(self.conv_1x3x3_5_1(x))))
        x = self.tail(torch.cat([x1, x3, x5], dim=1))
        # x = self.last(x)
        # x = torch.squeeze(x, dim=2)
        x = self.sigmoid(x[:, :, -1, :, :])
        out2 = x

        out = residual + residual * out1 * out2

        return out

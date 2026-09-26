import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from numpy import *
import numpy as np
import scipy.io as scio

#将单通道重复 3 次，模拟 3 通道输入

# # no-crop
# class TrainDataset(Dataset):
#     def __init__(self, dataset_cfg):
#         self.root = dataset_cfg['root']
#         txtpath = os.path.join(self.root, 'train.txt')
#         txt = np.loadtxt(txtpath, dtype=bytes).astype(str)
#         self.imgs_arr = txt
#         self.full_supervision = dataset_cfg['full_supervision']
#         self.train_mean = 105.4025
#         self.train_std = 26.6452
#
#     def __getitem__(self, index):
#         img_path_mix = os.path.join(self.root, self.imgs_arr[index])
#
#         # Read Mix
#         MixData_mat = scio.loadmat(img_path_mix)
#         MixData_Img = MixData_mat.get('Mix')
#         MixData_Img = MixData_Img.astype(np.float32)
#
#         # Read Label/Tgt
#         img_path_tgt = img_path_mix.replace('.mat', '.png')
#         if self.full_supervision:
#             img_path_tgt = img_path_tgt.replace('Mix', 'masks')
#             LabelData_Img = Image.open(img_path_tgt)
#             LabelData_Img = np.array(LabelData_Img, dtype=np.float32) / 255.0
#         else:
#             img_path_tgt = img_path_tgt.replace('Mix', 'masks_centroid')
#             LabelData_Img = Image.open(img_path_tgt)
#             LabelData_Img = np.array(LabelData_Img, dtype=np.float32) / 255.0
#
#         # Mix preprocess
#         MixData_Img = (MixData_Img - self.train_mean) / self.train_std
#         MixData = torch.from_numpy(MixData_Img)
#
#         MixData_out = torch.unsqueeze(MixData[-5:, :, :], 0)  # the last five frame
#         MixData_out = MixData_out.repeat(3, 1, 1, 1)
#
#         # get name
#         img_path_tgt_split = img_path_tgt.split('/')
#         name = img_path_tgt_split[-3] + '/' + img_path_tgt_split[-1]
#
#         [m_L, n_L] = np.shape(LabelData_Img)
#         if m_L == 384 and n_L == 384:
#             # Tgt preprocess
#             LabelData = torch.from_numpy(LabelData_Img)
#             TgtData_out = torch.unsqueeze(LabelData, 0)
#             return MixData_out, TgtData_out, name
#
#         else:
#             m_L, n_L = min(m_L, 384), min(n_L, 384)
#             # Tgt preprocess
#             [n, t, m_M, n_M] = shape(MixData_out)
#             m_M, n_M = min(m_M, 384), min(n_M, 384)
#
#             LabelData_Img_1 = np.zeros([384, 384])
#             LabelData_Img_1[0:m_L, 0:n_L] = LabelData_Img[0:m_L, 0:n_L]
#             LabelData = torch.from_numpy(LabelData_Img_1)
#             TgtData_out = torch.unsqueeze(LabelData, 0)
#             MixData_out_1 = torch.zeros([n, t, 384, 384])
#             MixData_out_1[0:n, 0:t, 0:m_M, 0:n_M] = MixData_out[0:n, 0:t, 0:m_M, 0:n_M]
#             return MixData_out_1, TgtData_out, name
#
#     def __len__(self):
#         return len(self.imgs_arr)
#
#
# class ValDataset(Dataset):
#     def __init__(self, dataset_cfg, partition='all'):
#         self.root = dataset_cfg['root']
#         txtpath = os.path.join(self.root, 'test.txt')
#         txt = np.loadtxt(txtpath, dtype=bytes).astype(str)
#
#         if partition == 'all':
#             pass
#         elif partition == 'lSCR':
#             test_seqs_lSCR = [47, 56, 59, 76, 92, 101, 105, 119]  # snr < 3
#             filtered_paths = []
#             for seq in test_seqs_lSCR:
#                 filtered_paths.extend([path for path in txt if f'Sequence{seq}' in path])
#             txt = np.array(filtered_paths)
#         elif partition == 'hSCR':
#             test_seqs_hSCR = [85, 86, 87, 88, 89, 90, 91, 93, 94, 95, 96, 97]  # snr > 3
#             filtered_paths = []
#             for seq in test_seqs_hSCR:
#                 filtered_paths.extend([path for path in txt if f'Sequence{seq}' in path])
#             txt = np.array(filtered_paths)
#         else:
#             raise ValueError(f"Invalid partition '{partition}'. It must be 'all', 'lSCR' or 'hSCR'.")
#
#         self.imgs_arr = txt
#         self.full_supervision = dataset_cfg['full_supervision']
#         self.train_mean = 105.4025
#         self.train_std = 26.6452
#
#     def __getitem__(self, index):
#         img_path_mix = os.path.join(self.root, self.imgs_arr[index])
#
#         # Read Mix
#         MixData_mat = scio.loadmat(img_path_mix)
#         MixData_Img = MixData_mat.get('Mix')
#         MixData_Img = MixData_Img.astype(np.float32)
#
#         # Read Label/Tgt
#         img_path_tgt = img_path_mix.replace('.mat', '.png')
#         if self.full_supervision:
#             img_path_tgt = img_path_tgt.replace('Mix', 'masks')
#             LabelData_Img = Image.open(img_path_tgt)
#             LabelData_Img = np.array(LabelData_Img, dtype=np.float32) / 255.0
#         else:
#             img_path_tgt = img_path_tgt.replace('Mix', 'masks_centroid')
#             LabelData_Img = Image.open(img_path_tgt)
#             LabelData_Img = np.array(LabelData_Img, dtype=np.float32) / 255.0
#
#         # Mix preprocess
#         MixData_Img = (MixData_Img - self.train_mean) / self.train_std
#         MixData = torch.from_numpy(MixData_Img)
#
#         MixData_out = torch.unsqueeze(MixData[-5:, :, :], 0)  # the last five frame
#         MixData_out = MixData_out.repeat(3, 1, 1, 1)
#
#         # get name
#         img_path_tgt_split = img_path_tgt.split('/')
#         name = img_path_tgt_split[-3] + '/' + img_path_tgt_split[-1]
#
#         [m_L, n_L] = np.shape(LabelData_Img)
#         if m_L == 384 and n_L == 384:
#             # Tgt preprocess
#             LabelData = torch.from_numpy(LabelData_Img)
#             TgtData_out = torch.unsqueeze(LabelData, 0)
#             return MixData_out, TgtData_out, name
#
#         else:
#             m_L, n_L = min(m_L, 384), min(n_L, 384)
#             # Tgt preprocess
#             [n, t, m_M, n_M] = shape(MixData_out)
#             m_M, n_M = min(m_M, 384), min(n_M, 384)
#
#             LabelData_Img_1 = np.zeros([384, 384])
#             LabelData_Img_1[0:m_L, 0:n_L] = LabelData_Img[0:m_L, 0:n_L]
#             LabelData = torch.from_numpy(LabelData_Img_1)
#             TgtData_out = torch.unsqueeze(LabelData, 0)
#             MixData_out_1 = torch.zeros([n, t, 384, 384])
#             MixData_out_1[0:n, 0:t, 0:m_M, 0:n_M] = MixData_out[0:n, 0:t, 0:m_M, 0:n_M]
#             return MixData_out_1, TgtData_out, name
#
#     def __len__(self):
#         return len(self.imgs_arr)


# crop
class TrainDataset(Dataset):
    def __init__(self, dataset_cfg):
        self.root = dataset_cfg['root']
        txtpath = os.path.join(self.root, 'train.txt')
        txt = np.loadtxt(txtpath, dtype=bytes).astype(str)
        self.imgs_arr = txt
        self.full_supervision = dataset_cfg['full_supervision']
        self.train_mean = 105.4025
        self.train_std = 26.6452

    def __getitem__(self, index):
        img_path_mix = os.path.join(self.root, self.imgs_arr[index])

        # Read Mix
        MixData_mat = scio.loadmat(img_path_mix)
        MixData_Img = MixData_mat.get('Mix')
        MixData_Img = MixData_Img.astype(np.float32)

        # Read Label/Tgt
        img_path_tgt = img_path_mix.replace('.mat', '.png')
        if self.full_supervision:
            img_path_tgt = img_path_tgt.replace('Mix', 'masks')
            LabelData_Img = Image.open(img_path_tgt)
            LabelData_Img = np.array(LabelData_Img, dtype=np.float32) / 255.0
        else:
            img_path_tgt = img_path_tgt.replace('Mix', 'masks_centroid')
            LabelData_Img = Image.open(img_path_tgt)
            LabelData_Img = np.array(LabelData_Img, dtype=np.float32) / 255.0

        # Mix preprocess
        MixData_Img = (MixData_Img - self.train_mean) / self.train_std
        MixData = torch.from_numpy(MixData_Img)

        MixData_out = torch.unsqueeze(MixData[-5:, :, :], 0)  # the last five frame 在第 0 维插入一个新轴，长度为 1
        MixData_out = MixData_out.repeat(3, 1, 1, 1)#会把张量在第0维重复3次、其余维度不变，

        # get name
        img_path_tgt_split = img_path_tgt.split('/')
        name = img_path_tgt_split[-3] + '/' + img_path_tgt_split[-1]

        [m_L, n_L] = np.shape(LabelData_Img)
        if m_L == 384 and n_L == 384:
            # Tgt preprocess
            LabelData = torch.from_numpy(LabelData_Img)
            TgtData_out = torch.unsqueeze(LabelData, 0)
            return MixData_out, TgtData_out, m_L, n_L, name

        else:
            m_L, n_L = min(m_L, 384), min(n_L, 384)
            # Tgt preprocess
            [n, t, m_M, n_M] = shape(MixData_out)
            m_M, n_M = min(m_M, 384), min(n_M, 384)

            LabelData_Img_1 = np.zeros([384, 384])
            LabelData_Img_1[0:m_L, 0:n_L] = LabelData_Img[0:m_L, 0:n_L]
            LabelData = torch.from_numpy(LabelData_Img_1)
            TgtData_out = torch.unsqueeze(LabelData, 0)
            MixData_out_1 = torch.zeros([n, t, 384, 384])
            MixData_out_1[0:n, 0:t, 0:m_M, 0:n_M] = MixData_out[0:n, 0:t, 0:m_M, 0:n_M]
            return MixData_out_1, TgtData_out, m_L, n_L, name
        #(3,5,384,384)
    def __len__(self):
        return len(self.imgs_arr)


class ValDataset(Dataset):
    def __init__(self, dataset_cfg, split='all'):
        self.root = dataset_cfg['root']
        txtpath = os.path.join(self.root, 'test.txt')
        txt = np.loadtxt(txtpath, dtype=bytes).astype(str)

        if split == 'all':
            pass
        elif split == 'lSCR':
            test_seqs_lSCR = [47, 56, 59, 76, 92, 101, 105, 119]  # scr < 1
            filtered_paths = []
            for seq in test_seqs_lSCR:
                filtered_paths.extend([path for path in txt if f'Sequence{seq}' in path])
            txt = np.array(filtered_paths)
        else:
            raise ValueError(f"Invalid split '{split}'. It must be 'all' or 'lSCR'.")

        self.imgs_arr = txt
        self.full_supervision = dataset_cfg['full_supervision']
        self.train_mean = 105.4025
        self.train_std = 26.6452

    def __getitem__(self, index):
        img_path_mix = os.path.join(self.root, self.imgs_arr[index])

        # Read Mix
        MixData_mat = scio.loadmat(img_path_mix)
        MixData_Img = MixData_mat.get('Mix')
        MixData_Img = MixData_Img.astype(np.float32)

        # Read Label/Tgt
        img_path_tgt = img_path_mix.replace('.mat', '.png')
        if self.full_supervision:
            img_path_tgt = img_path_tgt.replace('Mix', 'masks')
            LabelData_Img = Image.open(img_path_tgt)
            LabelData_Img = np.array(LabelData_Img, dtype=np.float32) / 255.0
        else:
            img_path_tgt = img_path_tgt.replace('Mix', 'masks_centroid')
            LabelData_Img = Image.open(img_path_tgt)
            LabelData_Img = np.array(LabelData_Img, dtype=np.float32) / 255.0

        # Mix preprocess
        MixData_Img = (MixData_Img - self.train_mean) / self.train_std
        MixData = torch.from_numpy(MixData_Img)

        MixData_out = torch.unsqueeze(MixData[-5:, :, :], 0)  # the last five frame
        MixData_out = MixData_out.repeat(3, 1, 1, 1)

        # get name
        img_path_tgt_split = img_path_tgt.split('/')
        name = img_path_tgt_split[-3] + '/' + img_path_tgt_split[-1]

        [m_L, n_L] = np.shape(LabelData_Img)
        if m_L == 384 and n_L == 384:
            # Tgt preprocess
            LabelData = torch.from_numpy(LabelData_Img)
            TgtData_out = torch.unsqueeze(LabelData, 0)
            return MixData_out, TgtData_out, m_L, n_L, name

        else:
            m_L, n_L = min(m_L, 384), min(n_L, 384)
            # Tgt preprocess
            [n, t, m_M, n_M] = shape(MixData_out)
            m_M, n_M = min(m_M, 384), min(n_M, 384)

            LabelData_Img_1 = np.zeros([384, 384])
            LabelData_Img_1[0:m_L, 0:n_L] = LabelData_Img[0:m_L, 0:n_L]
            LabelData = torch.from_numpy(LabelData_Img_1)
            TgtData_out = torch.unsqueeze(LabelData, 0)
            MixData_out_1 = torch.zeros([n, t, 384, 384])
            MixData_out_1[0:n, 0:t, 0:m_M, 0:n_M] = MixData_out[0:n, 0:t, 0:m_M, 0:n_M]
            return MixData_out_1, TgtData_out, m_L, n_L, name

    def __len__(self):
        return len(self.imgs_arr)

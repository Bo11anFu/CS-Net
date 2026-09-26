import os
import torch
import torch.utils.data as Data
import torchvision.transforms as transforms
from glob import glob
from csnet.utils import (rgb_loader, binary_loader, random_flip, random_crop, random_rotation, color_enhance,
                            random_peper)


def get_sequence_name(path):
    normalized_path = path.replace('\\', '/').rstrip('/')
    return '/'.join(normalized_path.rsplit('/', 2)[-2:])



#frame(3, 5, 348, 348) mask(1, 348, 348) uint8
class MISTDataset(Data.Dataset):
    def __init__(self, root, num_inputs=5, img_size=None, frame_padding=False, data_aug=None, split='train'):
        if split == 'train':
            # 78 sequences
            seq_list = [1, 2, 5, 6, 8, 10, 11, 14, 17, 18, 19, 21, 23, 25, 26, 27, 28, 29, 30, 31, 33, 34, 39, 42, 44,
                        45, 47, 51, 52, 53, 56, 58, 59, 60, 61, 62, 63, 64, 65, 67, 68, 69, 70, 72, 73, 74, 78, 79, 81,
                        82, 85, 88, 89, 92, 94, 97, 98, 100, 101, 102, 103, 106, 107, 109, 111, 112, 113, 114, 115, 119,
                        123, 126, 128, 129, 132, 136, 139, 140]
        elif split == 'val_all':
            # 42 sequences
            seq_list = [57]
            #seq_list = [15, 16, 46, 75, 83, 84, 86, 90, 93, 96]
            #seq_list = [104, 105, 110]
            #seq_list = [3, 7, 12, 15, 20, 24, 35, 37, 43, 46, 76, 77, 83, 84, 86, 93, 108, 116, 17, 124, 133, 134] # speed test 5~7
            #seq_list = [91, 121, 125]

            #seq_list = [3, 7, 12, 13, 15, 16, 20, 24, 35, 37, 38, 40, 43, 46, 49, 55, 66, 75, 76, 77, 80, 83, 84, 86,
            #              90, 91, 93, 95, 96, 104, 105, 108, 110, 116, 117, 121, 122, 124, 125, 133, 134, 138]
        elif split == 'val_hard':
            # 11 sequences
            seq_list = [3, 16, 24, 35, 46, 84, 90, 93, 96, 105, 121]
        else:
            raise ValueError(
                f"Invalid split '{split}'. It must be 'train', 'val_all' or 'val_hard'.")

        if img_size is None:
            img_size = [384, 384]
        self.img_size = img_size

        if data_aug is None:
            data_aug = {
                'random_flip': True,
                'random_crop': True,
                'random_rotation': True,
                'color_enhance': False,
                'random_peper': False
            }
        self.num_inputs = num_inputs
        self.data_aug = data_aug
        self.split = split
        self.grouped_frame_paths = []
        self.grouped_mask_paths = []

        # grouping
        for seq in seq_list:
            frame_paths = sorted(glob(os.path.join(root, 'image', str(seq), '*.png')))
            mask_paths = sorted(glob(os.path.join(root, 'mask', str(seq), '*.png')))
            num_frames = len(frame_paths)
            assert num_inputs <= num_frames, f"number of input frames '{num_inputs}' exceeds the total number."
            if frame_padding:
                for i in range(num_frames):
                    frame_list = []
                    for j in range(num_inputs - 1, -1, -1):
                        if i - j < 0:
                            frame_list.append(frame_paths[0])
                        else:
                            frame_list.append(frame_paths[i - j])
                    self.grouped_frame_paths.append(frame_list)
                    self.grouped_mask_paths.append(mask_paths[i])
            else:
                #从第num_inputs-1帧开始创建输入组，确保每个组都有完整的num_inputs帧
                #不进行填充输入组会比帧数少num_input-1个
                for i in range(num_inputs - 1, num_frames):#i取4--num_frames-1
                    frame_list = []
                    #j = 4, 3, 2, 1, 0
                    for j in range(num_inputs - 1, -1, -1):
                        frame_list.append(frame_paths[i - j])
                    #包含所有输入帧组的路径，每个元素是一个长度为num_inputs的列表,元素为frame_list
                    self.grouped_frame_paths.append(frame_list)
                    #包含每个输入帧组对应的目标掩码路径
                    self.grouped_mask_paths.append(mask_paths[i])

        # transforms
        self.frame_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.359, 0.359, 0.359], [0.15, 0.15, 0.15])])
        # self.frame_transform = transforms.Compose([
        #     transforms.Resize(img_size),
        #     transforms.Grayscale(num_output_channels=1),  # 转单通道
        #     transforms.ToTensor(),  # 自动除以255，像素值从[0,255]→[0,1]
        # ])
        self.mask_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor()])

    def __getitem__(self, index):
        frames = []
        for i in range(self.num_inputs):
            #从每个输入帧组中取出第i帧
            frames.append(rgb_loader(self.grouped_frame_paths[index][i]))
        mask = binary_loader(self.grouped_mask_paths[index])

        # data augmentation (only for training)
        if self.split == 'train':
            if self.data_aug['random_flip']:
                frames, mask = random_flip(frames, mask)
            if self.data_aug['random_crop']:
                frames, mask = random_crop(frames, mask)
            if self.data_aug['random_rotation']:
                frames, mask = random_rotation(frames, mask)
            if self.data_aug['color_enhance']:
                frames = color_enhance(frames)
            if self.data_aug['random_peper']:
                mask = random_peper(mask)

        # transforms
        for i in range(len(frames)):
            frames[i] = self.frame_transform(frames[i])
        frames = torch.stack(frames, dim=1)
        mask = self.mask_transform(mask)

        # get name
        #从列表中取出第 index 个路径字符串，用“/”作为分隔符切成列表，
        # 如['...','parent','file.png']
        name = get_sequence_name(self.grouped_mask_paths[index])

        # # 获取所有5帧的路径名（用于特征可视化）
        frame_names = []
        for i in range(self.num_inputs):
            frame_names.append(get_sequence_name(self.grouped_frame_paths[index][i]))

        # return frames, mask, name
        return frames, mask, self.img_size[0], self.img_size[1], name, frame_names


        #original code watch windows        # return frames, mask, name
        #return frames, mask, self.img_size[0], self.img_size[1], name
    def __len__(self):
        return len(self.grouped_frame_paths)

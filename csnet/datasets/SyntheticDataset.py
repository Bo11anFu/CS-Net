import os
import torch
import torch.utils.data as Data
import torchvision.transforms as transforms
from glob import glob
from PIL import Image
import tifffile
import numpy as np


#frame(3, 5, 2048, 2048) mask(1, 2048, 2048) uint12

#frame(3, 5, 348, 348) mask(1, 348, 348) uint12

class SyntheticDataset(Data.Dataset):
    def __init__(self, root, num_inputs=5, img_size=None, split='val_all'):
        if split == 'val_all':
            seq_list = [1]
        #elif split == 'val_hard':
            #seq_list = [2]
        else:
            raise ValueError(
                f"Invalid split '{split}'. It must be 'train', 'val_all' or 'val_hard'.")


        self.root = root
        self.num_inputs = num_inputs

        if img_size is None:
            img_size = [348, 348]
        self.img_size = img_size
        self.split = split
        self.grouped_frame_paths = []
        self.grouped_mask_paths = []

        for seq in seq_list:
            frame_paths = sorted(glob(os.path.join(root, 'img', str(seq), '*.tiff')))
            mask_paths = sorted(glob(os.path.join(root, 'mask', str(seq), '*.tiff')))
            num_frames = len(frame_paths)
            if not frame_paths or not mask_paths:
                raise FileNotFoundError(f"在 '{root}/img' 或 '{root}/mask' 目录下没有找到 .tiff 文件。请检查路径。")
            assert num_inputs <= num_frames, f"输入帧数 '{num_inputs}' 超过了总帧数 '{num_frames}'。"
            #从第num_inputs-1帧开始创建输入组，确保每个组都有完整的num_inputs帧
            #不进行填充输入组会比帧数少num_input-1个
            for i in range(num_inputs - 1, num_frames):
                frame_list = []
                for j in range(num_inputs - 1, -1, -1):
                    frame_list.append(frame_paths[i - j])
                #包含所有输入帧组的路径，每个元素是一个长度为num_inputs的列表
                self.grouped_frame_paths.append(frame_list)
                #包含每个输入帧组对应的目标掩码路径
                self.grouped_mask_paths.append(mask_paths[i])
        
        """
        输入 [frame[0], frame[1], frame[2], frame[3], frame[4]]， 期望的输出（标签）是 mask[4]
        """


        #transform
        self.frame_transform = transforms.Compose([
            transforms.ToTensor(),
            #计算每个通道的均值和方差
            transforms.Normalize(mean=[0.612, 0.612, 0.612], std=[0.09, 0.09, 0.09])])
        self.mask_transform = transforms.Compose([
            transforms.ToTensor()])
        # 获取图像尺寸
        sample_img = tifffile.imread(frame_paths[0])
        self.img_height, self.img_width = sample_img.shape
        
        print(f"✅ 自定义数据集加载完成：共 {len(self.grouped_frame_paths)} 组测试数据。")
        print(f"   图像尺寸自动检测为: {self.img_height}x{self.img_width}")
    def __getitem__(self, index):
        frames = []
        for i in range(self.num_inputs):
            #从每个输入帧组中取出第i帧
            #frames.append(self.grouped_frame_paths[index][i])

            img_path = self.grouped_frame_paths[index][i]
            img_12bit = tifffile.imread(img_path).astype(np.float32)

            #  将数值范围从 [0, 4095] 缩放到 [0.0, 1.0]
            img_scaled = img_12bit / 4095.0
            
            # 将单通道 (H, W) 堆叠为三通道 (H, W, 3)
            img_3channel = np.stack([img_scaled] * 3, axis=-1)
             
            # transform.ToTensor() 会将 (H, W, 3) -> (3, H, W)
            # transforms.Normalize 期望接收到的输入张量（input）的数值范围已经是 [0.0, 1.0]
            frames.append(self.frame_transform(img_3channel))
            #堆叠形状为 (3, num_inputs, H, W)

        frames_tensor = torch.stack(frames, dim=1)
        # mask = self.mask_transform(tifffile.imread(self.grouped_mask_paths[index]))
        # mask_processed = (mask > 0).astype(np.uint16) * 65535
        # mask_tensor = torch.from_numpy(mask_processed)
        # mask_tensor = mask_tensor.unsqueeze(0)

        mask_path = self.grouped_mask_paths[index]
        mask_uint16 = tifffile.imread(mask_path)

        #布尔比较 + astype，将mask变为二进制，1表示目标区域，0表示背景
        mask_float32 = (mask_uint16 > 0).astype(np.float32)
        mask_tensor = self.mask_transform(mask_float32)

        # get name
        mask_path_split = self.grouped_mask_paths[index].split('/')
        name = mask_path_split[-2] + '/' + mask_path_split[-1]
        return frames_tensor, mask_tensor, self.img_height, self.img_width, name
    
    def __len__(self):
        return len(self.grouped_frame_paths)
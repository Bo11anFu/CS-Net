import os

from data_utils import *
import matplotlib.pyplot as plt
import json
from torchvision import transforms
import torch.nn.functional as F
import cv2

def seed_pytorch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def draw_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = round(min(x, radius)), round(min(width - x, radius + 1))
    top, bottom = round(min(y, radius)), round(min(height - y, radius + 1))

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap



def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

class TrainSetLoader(Dataset):
    def __init__(self, cfg):
        super(TrainSetLoader).__init__()
        self.trainset_dir = cfg.train_dataset_dir
        self.trainlabel_dir = cfg.train_label_dir
        self.input_num = cfg.input_num
        self.video_list = os.listdir(cfg.train_dataset_dir)

        file_count = 0
        for root, dirs, files in os.walk(self.trainset_dir):
            # for dir in dirs:
            #     dir_count += 1  # 统计文件夹下的文件夹总个数
            for _ in files:
                file_count += 1  # 统计文件夹下的文件总个数
        self.file_count = file_count
        self.n_iters = self.file_count # // cfg.batch_size

    def __getitem__(self, idx):
        self.video_list.sort()
        idx_video = random.randint(0, len(self.video_list) - 1)
        idx_frame = random.randint(self.input_num - 1,
                                   len(os.listdir(self.trainset_dir + '/' + self.video_list[idx_video])) - 1)
        gt = []
        input = []
        # n = 3
        radius = 5
        for i in range(idx_frame - self.input_num + 1, idx_frame + 1):
            img = Image.open(self.trainset_dir + '/' + self.video_list[idx_video] + '/' + str(i) + '.png')
            frame_json = json.load(open(self.trainlabel_dir + '/' + self.video_list[idx_video] + '.json'))
            f_dict = frame_json[i]
            num_obj = f_dict["num_objects"]
            img_gt = np.zeros((512, 640))
            for ii in range(num_obj):
                x = int(f_dict["object_coords"][ii][0])
                y = int(f_dict["object_coords"][ii][1])
                # img_gt[y - n // 2:y + n // 2, x - n // 2:x + n // 2] = 255
                ct_int = np.array(f_dict["object_coords"][ii], dtype=np.int32)
                img_gt = draw_gaussian(img_gt, ct_int, radius)
            img_gt = np.array(img_gt, dtype=np.float32) / 255
            img_input = np.array(img, dtype=np.float32) /255 # .squeeze()
            gt.append(img_gt)
            input.append(img_input)

        gt = np.stack(gt, 0)
        input = np.stack(input, 0)
        gt, input = random_crop(gt, input)
        gt, input = augumentation(gt, input)
        gt = torch.from_numpy(np.ascontiguousarray(gt.copy())).unsqueeze(1)
        input = torch.from_numpy(np.ascontiguousarray(input.copy())).unsqueeze(1)

        return input, gt

    def __len__(self):
        return self.n_iters

class TestSetLoader(Dataset):
    def __init__(self, dataset_dir, label_dir, video_name, input_num):
        super(TestSetLoader).__init__()
        self.dataset_dir = dataset_dir
        self.label_dir = label_dir
        self.img_list = os.listdir(self.dataset_dir)
        self.totensor = transforms.ToTensor()
        self.v_num = video_name
        self.input_num = input_num
    def __getitem__(self, idx):
        gt = []
        input = []
        n=3
        radius = 5
        for idx_frame in range(idx - self.input_num + 1, idx + 1):
            img = Image.open(self.dataset_dir + '/' + str(idx_frame) + '.png')
            frame_json = json.load(open(self.label_dir + '/' + self.v_num + '.json'))
            f_dict = frame_json[idx_frame]
            num_obj = f_dict["num_objects"]
            img_gt = np.zeros((512, 640))
            for ii in range(num_obj):
                x = int(f_dict["object_coords"][ii][0])
                y = int(f_dict["object_coords"][ii][1])
                # img_gt[y - n // 2:y + n // 2, x - n // 2:x + n // 2] = 255
                ct_int = np.array(f_dict["object_coords"][ii], dtype=np.int32)
                img_gt = draw_gaussian(img_gt, ct_int, radius)
            img_gt = np.array(img_gt, dtype=np.float32) / 255
            img_input = np.array(img, dtype=np.float32) / 255 # .squeeze()
            gt.append(img_gt)
            input.append(img_input)

        gt = np.stack(gt, 0)
        input = np.stack(input, 0)
        gt, input = random_crop(gt, input)
        gt = torch.from_numpy(np.ascontiguousarray(gt.copy())).unsqueeze(1)
        input = torch.from_numpy(np.ascontiguousarray(input.copy())).unsqueeze(1)

        return input, gt

    def __len__(self):
        return len(self.img_list)

import random
from PIL import Image

def random_crop(gt, input):
    n, h, w = input.shape
    crop_size = (256, 256)
    assert h >= crop_size[0] and w >= crop_size[1], "Crop size is larger than input size"

    top = np.random.randint(0, h - crop_size[0] + 1)
    left = np.random.randint(0, w - crop_size[1] + 1)

    input_cropped = input[:, top:top + crop_size[0], left:left + crop_size[1]]
    gt_cropped = gt[:, top:top + crop_size[0], left:left + crop_size[1]]

    return gt_cropped, input_cropped

def augumentation(input, target):
    if random.random() < 0.5:
        input = input[:, :, :, ::-1]
        target = target[:, :, :, ::-1]
    elif random.random() < 0.5:
        input = input[:, :, ::-1, :]
        target = target[:, :, ::-1, :]
    # elif random.random() < 0.5:
    #     input = input.transpose(0, 1, 3, 2)#C N H W
    #     target = target.transpose(0, 1, 3, 2)
    return input, target

class DSATtrain(Dataset):
    def __init__(self, cfg):
        super(TrainSetLoader).__init__()
        self.trainset_dir = cfg.train_dataset_dir
        self.trainlabel_dir = cfg.train_label_dir
        self.input_num = cfg.input_num
        self.video_list = os.listdir(cfg.train_dataset_dir)

        file_count = 0
        for root, dirs, files in os.walk(self.trainset_dir):
            # for dir in dirs:
            #     dir_count += 1  # 统计文件夹下的文件夹总个数
            for _ in files:
                file_count += 1  # 统计文件夹下的文件总个数
        self.file_count = file_count
        self.n_iters = self.file_count # // cfg.batch_size

    def __getitem__(self, idx):
        self.video_list.sort()
        idx_video = random.randint(0, len(self.video_list)-1)
        idx_frame = random.randint(self.input_num-1, len(os.listdir(self.trainset_dir + '/' + self.video_list[idx_video]))-1)
        gt = []
        input = []
        for i in range(idx_frame-self.input_num + 1, idx_frame+1):
            # img = Image.open(self.trainset_dir + '/' + self.video_list[idx_video] + '/' + str(i) + '.bmp')
            img = cv2.imread(self.trainset_dir + '/' + self.video_list[idx_video] + '/' + str(i) + '.bmp', cv2.IMREAD_GRAYSCALE)
            # img = np.array(img, dtype=np.float32)
            # img = Normalized(np.array(img, dtype=np.float32), img_norm_cfg=dict(mean=127.3055648803711, std=27.67917823791504))
            frame_json = json.load(open(self.trainlabel_dir + '/' + self.video_list[idx_video] + '.json'))
            f_dict = frame_json[i]
            num_obj = f_dict["num_objects"]
            img_gt = np.zeros((256, 256))
            radius = 7
            n = 5
            for ii in range(num_obj):
                x = int(f_dict["object_coords"][ii][0])
                y = int(f_dict["object_coords"][ii][1])
                # img_gt[y - n // 2:y + n // 2, x - n // 2:x + n // 2] = 1
                ct_int = np.array(f_dict["object_coords"][ii], dtype=np.int32)
                img_gt = draw_gaussian(img_gt, ct_int, radius)

            img_gt = np.array(img_gt * 255, dtype=np.float32)
            img = F.to_pil_image(img)
            img_gt = F.to_pil_image(img_gt)
            # img, img_gt = transform(img, img_gt)
            img_gt = np.array(img_gt, dtype=np.float32) / 255
            img = np.array(img, dtype=np.float32) / 255

            img_gt = img_gt[np.newaxis, :]
            img_input = img[np.newaxis, :]
            # img_gt = np.array(img_gt, dtype=np.float32)
            # img_input = np.array(img_input, dtype=np.float32)#.squeeze()
            # gt.append(img_gt)
            # input.append(img_input)

            gt.append(img_gt)
            input.append(img_input)

        gt = np.stack(gt, 0)
        input = np.stack(input, 0)
        input, gt = augumentation(input, gt)
        gt = torch.from_numpy(np.ascontiguousarray(gt.copy()))
        input = torch.from_numpy(np.ascontiguousarray(input.copy()))
        # input = bg_aligned(input)

        return input, gt

    def __len__(self):
        return self.n_iters

class DSATtest(Dataset):
    def __init__(self, dataset_dir, label_dir, video_name, input_num):
        super(TestSetLoader).__init__()
        self.dataset_dir = dataset_dir
        self.label_dir = label_dir
        self.img_list = os.listdir(self.dataset_dir)
        self.totensor = transforms.ToTensor()
        self.v_num = video_name
        self.input_num = input_num


    def __getitem__(self, idx):
        gt = []
        input = []
        # idx = random.randint(self.input_num-1, len(os.listdir(self.dataset_dir))-1)
        #for idx in range(self.input_num-1, len(os.listdir(self.dataset_dir))-1):
        for idx_frame in range(idx-self.input_num+1, idx+1):
            img = cv2.imread(self.dataset_dir + '/' + str(idx_frame) + '.bmp', cv2.IMREAD_GRAYSCALE)
            # img = np.array(img, dtype=np.float32)
            # img = Normalized(np.array(img, dtype=np.float32), img_norm_cfg=dict(mean=127.3055648803711, std=27.67917823791504))
            frame_json = json.load(open(self.label_dir + '/' + self.v_num+'.json'))
            f_dict = frame_json[idx_frame]
            num_obj = f_dict["num_objects"]
            img_gt = np.zeros((256, 256))
            radius = 7
            n = 5
            for ii in range(num_obj):
                x = int(f_dict["object_coords"][ii][0])
                y = int(f_dict["object_coords"][ii][1])
                # img_gt[y - n // 2:y + n // 2, x - n // 2:x + n // 2] = 1
                ct_int = np.array(f_dict["object_coords"][ii], dtype=np.int32)
                img_gt = draw_gaussian(img_gt, ct_int, radius)

            img_gt = np.array(img_gt * 255, dtype=np.float32)
            img = F.to_pil_image(img)
            img_gt = F.to_pil_image(img_gt)
            img_gt = np.array(img_gt, dtype=np.float32) / 255
            img = np.array(img, dtype=np.float32) / 255  # .squeeze()

            img_gt = img_gt[np.newaxis, :]
            img_input = img[np.newaxis, :]


            gt.append(img_gt)
            input.append(img_input)

        gt = np.stack(gt, 0)
        input = np.stack(input, 0)


        gt = torch.from_numpy(np.ascontiguousarray(gt.copy()))
        input = torch.from_numpy(np.ascontiguousarray(input.copy()))

        return input, gt

    def __len__(self):
        return len(self.img_list)

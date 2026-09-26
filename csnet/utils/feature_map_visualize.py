import cv2
import numpy as np
import os
import torch
import matplotlib.pyplot as plt

from csnet.utils.data_processing import rgb_loader


def featuremap_2_heatmap(feature_map):
    assert isinstance(feature_map, torch.Tensor)
    feature_map = feature_map.detach()
    heatmap = feature_map[:, 0, :, :] * 0
    for c in range(feature_map.shape[1]):
        heatmap += feature_map[:, c, :, :]
    heatmap = heatmap.cpu().numpy()
    heatmap = np.mean(heatmap, axis=0)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    return heatmap


def draw_feature_map(feature_maps, img_path='', save_dir='', name=None):
    # img = rgb_loader(img_path)  # 测试用PIL还是cv2
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if not isinstance(feature_maps, (list, tuple)):
        feature_maps = [feature_maps]
    for idx, feature_map in enumerate(feature_maps):
        heatmap = featuremap_2_heatmap(feature_map)
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = heatmap * 0.5 + img * 0.3
        cv2.imwrite(os.path.join(save_dir, name + '_' + str(idx) + '.png'), superimposed_img)
        # plt.imshow(superimposed_img)
        # plt.show()

        # cv2.imshow("1", superimposed_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


def draw_single_feature_map(feature_map, img_path, save_path):
    """绘制单张特征图热力图叠加原图并保存

    Args:
        feature_map: torch.Tensor, 特征图
        img_path: str, 原图路径
        save_path: str, 保存路径
    """
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"Warning: cannot read image from {img_path}")
        return
    heatmap = featuremap_2_heatmap(feature_map)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.5 + img * 0.3
    cv2.imwrite(save_path, superimposed_img)


def draw_feature_maps_batch(feature_maps_list, img_path, save_dir, name_prefix=''):
    """批量绘制多个特征图热力图叠加原图

    Args:
        feature_maps_list: list of torch.Tensor, 特征图列表
        img_path: str, 原图路径
        save_dir: str, 保存目录
        name_prefix: str, 文件名前缀
    """
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"Warning: cannot read image from {img_path}")
        return
    for idx, feature_map in enumerate(feature_maps_list):
        heatmap = featuremap_2_heatmap(feature_map)
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = heatmap * 0.5 + img * 0.3
        cv2.imwrite(os.path.join(save_dir, name_prefix + str(idx) + '.png'), superimposed_img)

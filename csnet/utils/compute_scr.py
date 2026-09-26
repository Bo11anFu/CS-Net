#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os


def getSCR(img, targetMask, pos, factor):
    x, y, w, h = pos
    height, width = img.shape

    tmpImg = img.copy()

    roi = tmpImg[y:y + h, x:x + w]
    true_tg_mean = np.mean(roi[targetMask == 255])

    tmp_rec = np.array([y - factor, y + h + factor, x - factor, x + w + factor])
    tmp_rec = np.where(tmp_rec > 0, tmp_rec, 0)  # tmp_rec为四个坐标值
    tmp_rec[1] = min(height - 1, tmp_rec[1])
    tmp_rec[3] = min(width - 1, tmp_rec[3])

    roi[targetMask == 255] = 0
    tmpImg[y:y + h, x:x + w] = roi

    backImg = tmpImg[tmp_rec[0]:tmp_rec[1], tmp_rec[2]:tmp_rec[3]]

    # mask = np.zeros((tmp_rec[1]-tmp_rec[0],tmp_rec[3]-tmp_rec[2]), dtype=int)
    # mask[factor:factor + h, factor:factor + w] = targetMask

    # backImg[mask == 255] = 0
    backImg = backImg.flatten()

    zero_index = np.where(backImg == 0)
    backImg = np.delete(backImg, zero_index)

    bc_mean = np.mean(backImg)

    bc_std = np.std(backImg, ddof=0)
    bc_std = max(1, bc_std)

    S = true_tg_mean - bc_mean
    SCR = S / (bc_std + 1e-5)
    return abs(SCR)


def minRect(img):
    _, thresh = cv2.threshold(img, 90, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])
    return x, y, w, h


if __name__ == "__main__":

    base_path = r'/media/estar/Data1/gr/Deep-MIRST-main/data/SHU-MIRST-0124'

    img_paths = os.path.join(base_path, 'image')

    mask_paths = os.path.join(base_path, 'mask')

    folders = os.listdir(img_paths)
    folders.sort(key=lambda path: int(path))

    for folder in folders:
        scr_list = []
        folder_path = os.path.join(img_paths, folder)
        length = len(os.listdir(folder_path))
        for img_path in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_path)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            mask_path = img_path.replace('image', 'mask')
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            _, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)

            try:
                x, y, w, h = minRect(mask)
                SCR = getSCR(img, mask[y:y + h, x:x + w], (x, y, w, h), 20)
            except IndexError:
                SCR = 0

            scr_list.append(SCR)

        scr_list = np.array(scr_list)
        print(f'seq{folder} mSCR: {scr_list.mean():.2f}  std: {scr_list.std():.2f}')

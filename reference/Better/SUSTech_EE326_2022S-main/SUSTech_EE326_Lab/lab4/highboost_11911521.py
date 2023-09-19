#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :highboost_11911521.py
# @Time      :2022-03-25 19:37
# @Author    :钟新宇
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


def highboost(img, k=1):
    m, n = img.shape
    mean_img = cv.blur(img, (3, 3))

    g_mask = np.array(img - mean_img, dtype=int)

    out_img = np.array(img + k * g_mask, dtype=int)
    out_img = np.array(255 * np.divide(out_img, max(out_img.flat)), dtype=int)
    for i in range(m):
        for j in range(n):
            if out_img[i, j] < 0: out_img[i, j] = 0

    return out_img, g_mask


if __name__ == '__main__':
    try:
        img = np.array(cv.imread("./Q4_1.tif", cv.IMREAD_GRAYSCALE), dtype=int)
        [unsharped_img, unsharped_mask] = highboost(img, 1)

        plt.figure()
        plt.subplot(121)
        plt.title('unsharped_img')
        plt.imshow(unsharped_img, cmap='gray')
        plt.subplot(122)
        plt.title('unsharped_mask')
        plt.imshow(unsharped_mask, cmap='gray')
        plt.show()
    except KeyboardInterrupt:
        pass

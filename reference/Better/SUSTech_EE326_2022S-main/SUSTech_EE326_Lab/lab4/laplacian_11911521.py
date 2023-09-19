#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :laplacian_11911521.py
# @Time      :2022-03-25 19:07
# @Author    :钟新宇

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

operator1 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], int)
operator2 = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], int)
operator3 = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], int)
operator4 = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], int)


def laplacian(img, operator_type="operator1", k=1):
    global operator
    if operator_type == "operator1":
        operator = operator1
    elif operator_type == "operator2":
        operator = operator2
    elif operator_type == "operator3":
        operator = operator3
    elif operator_type == "operator4":
        operator = operator4
    else:
        pass

    m, n = img.shape
    laplacian_img = np.zeros((m, n), int)
    size = 1

    pad_img = np.pad(img, size)

    for i in range(m):
        for j in range(n):
            neighbor_img = pad_img[i:i + 2 * size + 1, j:j + 2 * size + 1]
            tmp = np.multiply(neighbor_img, operator)
            tmp_sum = np.sum(tmp)
            # laplacian_img[i][j] = np.round(tmp_sum, int)
            laplacian_img[i][j] = tmp_sum

    out_img = np.array(img + k * laplacian_img, int)
    out_img = np.array(255 * np.divide(out_img, max(out_img.flat)), dtype=int)
    for i in range(m):
        for j in range(n):
            if out_img[i, j] < 0: out_img[i, j] = 0
    return out_img, laplacian_img


if __name__ == '__main__':
    try:
        file = "./Q4_1.tif"
        img = np.array(cv.imread(file, cv.IMREAD_GRAYSCALE), int)

        [laplacian_img, laplacian_mask] = laplacian(img, "operator1", 1)

        plt.figure()
        plt.subplot(121)
        plt.title('laplacian image')
        plt.imshow(laplacian_img, cmap='gray')
        plt.subplot(122)
        plt.title('laplacian_mask')
        plt.imshow(laplacian_mask, cmap='gray')
        plt.show()
    except KeyboardInterrupt:
        pass

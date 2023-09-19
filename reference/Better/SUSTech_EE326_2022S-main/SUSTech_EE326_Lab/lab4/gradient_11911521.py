#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :gradient_11911521.py
# @Time      :2022-03-25 19:42
# @Author    :钟新宇
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

Roberts = (np.array([[-1, 0], [0, 1]], int), np.array([[0, -1], [1, 0]], int))
Sobel = (np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], int), np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], int))


def gradient(img, operator_type="sobel"):
    global operator, size
    m, n = img.shape
    gradient_img = np.zeros((m, n), int)

    if operator_type == "sobel":
        operator = Sobel
        size = 3
    elif operator_type == "roberts":
        operator = Roberts
        size = 2
    else:
        pass

    pad_img = np.pad(img, size)
    for i in range(m):
        for j in range(n):
            neighbor_img = pad_img[i:i + size, j:j + size]

            gradient_x = np.sum(operator[0] * neighbor_img)
            gradient_y = np.sum(operator[1] * neighbor_img)
            gradient_sum = np.abs(gradient_x) + np.abs(gradient_y)
            gradient_img[i, j] = gradient_sum

    gradient_img = np.array(255 * np.divide(gradient_img, max(gradient_img.flat)), dtype=int)
    out_img = np.array(img + gradient_img, dtype=int)
    out_img = np.array(255 * np.divide(out_img, max(out_img.flat)), dtype=int)
    for i in range(m):
        for j in range(n):
            if out_img[i, j] < 0: out_img[i, j] = 0

    return out_img, gradient_img


if __name__ == '__main__':
    try:
        file = './Q4_1.tif'
        [roberts_img, roberts_mask] = gradient(cv.imread(file, cv.IMREAD_GRAYSCALE), "roberts")

        plt.figure()
        plt.subplot(121)
        plt.title('roberts_img')
        plt.imshow(roberts_img, cmap='gray')
        plt.subplot(122)
        plt.title('roberts_mask')
        plt.imshow(roberts_mask, cmap='gray')
        plt.show()
    except KeyboardInterrupt:
        pass

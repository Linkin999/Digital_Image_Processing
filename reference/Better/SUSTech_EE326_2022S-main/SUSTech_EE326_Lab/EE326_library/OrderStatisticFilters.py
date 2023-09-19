#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :OrderStatisticFilters.py
# @Time      :2022-04-23 11:57
# @Author    :钟新宇
import numpy as np

from EE326_library.Base import normalize


def median_filter(input_image, size):
    """
    median filter 对椒盐噪声特别有效，并且不会导致图像边缘变模糊，也不会让图像形状大小改变。
    median filter 对均匀噪声无效。
    :param input_image: 使用opencv读取的输入图像数组
    :param size: 邻域大小，正方形
    :return: 输出图像数组
    """
    img = np.array(input_image, dtype=float)
    row, col = img.shape
    step = (size - 1) // 2
    if step * 2 + 1 >= row or step * 2 + 1 >= col:
        print("The parameter size is to large.")
    img_pad = np.pad(input_image, [step, step], 'constant', constant_values=[1] * 2)
    img_out = np.array(np.zeros((row, col)))
    for i in range(step, row):
        for j in range(step, col):
            img_out[i - step, j - step] = np.median(img_pad[i - step:i + step + 1, j - step:j + step + 1])
    img_out = normalize(img_out)
    return img_out


def max_filter(input_image, size):
    """
    max filter 适用于处理椒（pepper）噪声，但是它会导致图像中黑色区域变小，白色区域变大
    :param input_image:
    :param size:
    :return:
    """
    img = np.array(input_image, dtype=float)
    row, col = img.shape
    step = (size - 1) // 2
    if step * 2 + 1 >= row or step * 2 + 1 >= col:
        print("The parameter size is to large.")
    img_pad = np.pad(input_image, [step, step], 'constant', constant_values=[1] * 2)
    img_out = np.array(np.zeros((row, col)))
    for i in range(step, row):
        for j in range(step, col):
            img_out[i - step, j - step] = np.max(img_pad[i - step:i + step + 1, j - step:j + step + 1])
    img_out = normalize(img_out)
    return img_out


def min_filter(input_image, size):
    """
    min filter 适用于处理盐（salt）噪声，但是它会导致图像中白色区域变小，黑色区域变大
    :param input_image:
    :param size:
    :return:
    """
    img = np.array(input_image, dtype=float)
    row, col = img.shape
    step = (size - 1) // 2
    if step * 2 + 1 >= row or step * 2 + 1 >= col:
        print("The parameter size is to large.")
    img_pad = np.pad(input_image, [step, step], 'constant', constant_values=[0] * 2)
    img_out = np.array(np.zeros((row, col)))
    for i in range(step, row):
        for j in range(step, col):
            img_out[i - step, j - step] = np.min(img_pad[i - step:i + step + 1, j - step:j + step + 1])
    return img_out


def midpoint_filter(input_image, size):
    img = np.array(input_image, dtype=float)
    row, col = img.shape
    step = (size - 1) // 2
    if step * 2 + 1 >= row or step * 2 + 1 >= col:
        print("The parameter size is to large.")
    img_pad = np.pad(input_image, [step, step], 'constant', constant_values=[1] * 2)
    img_out = np.array(np.zeros((row, col)))
    for i in range(step, row):
        for j in range(step, col):
            temp_max = np.max(img_pad[i - step:i + step + 1, j - step:j + step + 1])
            temp_min = np.min(img_pad[i - step:i + step + 1, j - step:j + step + 1])
            img_out[i - step, j - step] = (temp_max + temp_min) / 2
    img_out = normalize(img_out)
    return img_out


def alpha_trimmed_mean(input_image, d, size):
    """
    α－裁剪均值滤波器
    修正阿尔法均值滤波器在邻域中，删除 d 个最低灰度值和 d 个最高灰度值，计算剩余像素的算术平均值作为输出结果
    :param input_image:
    :param d:
    :param size:
    :return:
    """
    img = np.array(input_image, dtype=float)
    row, col = img.shape
    step = (size - 1) // 2
    if step * 2 + 1 >= row or step * 2 + 1 >= col:
        print("The parameter size is to large.")
    if d >= size ** 2:
        print("The parameter d is to large.")
    img_pad = np.pad(input_image, [step, step], 'constant', constant_values=[1] * 2)
    img_out = np.array(np.zeros((row, col)))
    for i in range(step, row):
        for j in range(step, col):
            temp = np.sort(img_pad[i - step:i + step + 1, j - step:j + step + 1].flat)
            temp = np.sum(temp[d: size ** 2 - d])
            img_out[i - step, j - step] = temp / (size ** 2 - d * 2)
    img_out = normalize(img_out)
    return img_out


if __name__ == '__main__':
    try:
        pass
    except KeyboardInterrupt:
        pass

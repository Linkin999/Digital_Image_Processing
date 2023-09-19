#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :AdaptiveFilters.py
# @Time      :2022-04-23 11:57
# @Author    :钟新宇
import numpy as np

from EE326_library.Base import normalize
from EE326_library.MeanFilters import geometric_mean


def adaptive_arithmetic_mean(input_image, noise_var, size):
    """
    adaptive mean filter 相当于原图像和算数平均滤波的加权平均，权重由方差决定。
    注意：由于输入图像 == 原图像 + 噪声，因此邻域方差 >= 全局方差。
    1.全局方差 == 0，输出原图像。
    2.邻域方差 == 全局方差，输出算术平均。
    3.邻域方差 >> 全局方差，说明邻域中包含图像的有效信息，输出图像应当接近原图像（全局方差较小，接近原图像；全局方差较大，接近算术平均）。
    :param noise_var:
    :param input_image:
    :param size:
    :return:
    """
    img = np.array(input_image, dtype=int)
    row, col = img.shape
    img_out = np.array(np.zeros((row, col)))

    step = (size - 1) // 2
    if step * 2 + 1 >= row or step * 2 + 1 >= col:
        print("The parameter size is to large.")

    if noise_var == 0:
        img_out = img
    else:
        img_pad = np.pad(input_image, [step, step], 'constant', constant_values=[0] * 2)
        for i in range(step, row):
            for j in range(step, col):
                temp = img_pad[i - step:i + step, j - step:j + step]
                temp_var = np.var(temp)
                if noise_var == temp_var:
                    img_out[i - step, j - step] = np.mean(temp)
                elif temp_var == 0:
                    img_out[i - step, j - step] = img[i - step, j - step]
                else:
                    rat = noise_var / temp_var
                    val = img[i - step, j - step]
                    img_out[i - step, j - step] = (val - rat * (val - np.mean(temp)))

    img_out = normalize(img_out)
    return img_out


def adaptive_geometric_mean(input_image, noise_var, size):
    """
    adaptive mean filter 相当于原图像和算数平均滤波的加权平均，权重由方差决定。
    注意：由于输入图像 == 原图像 + 噪声，因此邻域方差 >= 全局方差。
    1.全局方差 == 0，输出原图像。
    2.邻域方差 == 全局方差，输出算术平均。
    3.邻域方差 >> 全局方差，说明邻域中包含图像的有效信息，输出图像应当接近原图像（全局方差较小，接近原图像；全局方差较大，接近算术平均）。
    :param noise_var:
    :param input_image:
    :param size:
    :return:
    """
    img = np.array(input_image, dtype=int)
    row, col = img.shape
    img_out = np.array(np.zeros((row, col)))

    step = (size - 1) // 2
    if step * 2 + 1 >= row or step * 2 + 1 >= col:
        print("The parameter size is to large.")

    if noise_var == 0:
        img_out = img
    else:
        img_pad = np.pad(input_image, [step, step], 'constant', constant_values=[0] * 2)
        for i in range(step, row):
            for j in range(step, col):
                temp = img_pad[i - step:i + step, j - step:j + step]
                temp_var = np.var(temp)
                img_out[i - step, j - step] = img[i - step, j - step] - (noise_var / temp_var) * (
                        img[i - step, j - step] - geometric_mean(temp, size))
    img_out = normalize(img_out)
    return img_out


def _adaptive_median_mask(s, img, i, j):
    row, col = img.shape
    step = (s - 1) // 2

    if step <= i <= row - step and step <= j <= col - step:
        temp = img[i - step:i + step + 1, j - step:j + step + 1]
        zmed = np.median(temp)
        zmax = np.max(temp)
        zmin = np.min(temp)
        zxy = img[i, j]
        a1 = zmed - zmin
        a2 = zmed - zmax
        b1 = zxy - zmin
        b2 = zxy - zmax
        return temp, zmed, zmax, zmin, zxy, a1, a2, b1, b2


def adaptive_median_filter(input_image, smax, smin):
    """
        adaptive median filter 适用于椒盐噪声，可以尽可能确保输出值不是脉冲
        1. a1 > 0 and a2 < 0:
            通过比较邻域内中值和最大值、最小值的关系判断中值是不是脉冲；
            如果条件满足，说明不是脉冲，goto State B；
            如果是脉冲，增加窗口大小；
            如果窗口增加到最大，中值还是一个脉冲，那么直接输出中值
        2.b1 > 0 and b2 < 0:
            通过比较原图像素点和邻域最大值、最小值的关系判断正在处理的点是不是脉冲
            如果条件满足，说明不是脉冲，输出原像素点的灰度值
            如果是脉冲，输出邻域中值（不是脉冲），相当于中值滤波
    :param input_image:
    :param smax:窗口最大值
    :param smin:窗口初始值
    :return:
    """
    img = np.array(input_image, dtype=int)
    row, col = img.shape
    img_out = np.array(np.zeros((row, col)))

    for i in range(row):
        for j in range(col):

            s = smin
            temp, zmed, zmax, zmin, zxy, a1, a2, b1, b2 = _adaptive_median_mask(s, img, i, j)
            while temp is not None:

                # if A1>0 and A2<0, go to stage B
                if a1 > 0 and a2 < 0:
                    # temp, zmed, zmax, zmin, zxy, _, _, b1, b2 = adaptive_median_mask(s - 2, img, i, j)
                    # if A1>0 and A2<0, output zxy
                    if b1 > 0 and b2 < 0:
                        img_out[i, j] = zxy
                        break
                    # else output zmed
                    else:
                        img_out[i, j] = zmed
                        break
                # else increase the window size
                else:
                    s += 2
                    # if window size s > smax, output zmed
                    if s > smax:
                        img_out[i, j] = zmed
                        break
    img_out = normalize(img_out)
    return img_out


if __name__ == '__main__':
    try:
        pass
    except KeyboardInterrupt:
        pass

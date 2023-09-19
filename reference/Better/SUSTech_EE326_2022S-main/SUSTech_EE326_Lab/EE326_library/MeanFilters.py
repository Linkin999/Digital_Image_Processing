#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :MeanFilters.py
# @Time      :2022-04-23 11:56
# @Author    :钟新宇
import numpy as np

from EE326_library.Base import normalize


def arithmetic_mean(input_image, size):
    img = np.array(input_image, dtype=int)
    row, col = img.shape
    step = (size - 1) // 2
    if step * 2 + 1 >= row or step * 2 + 1 >= col:
        print("The parameter size is to large.")
    img_pad = np.pad(input_image, [step, step], 'constant', constant_values=[0] * 2)
    img_out = np.array(np.zeros((row, col)))
    for i in range(step, row):
        for j in range(step, col):
            img_out[i - step, j - step] = np.sum(img_pad[i - step:i + step + 1, j - step:j + step + 1]) / (step ** 2)
    img_out = normalize(img_out)
    return img_out


def geometric_mean(input_image, size):
    """
    一般来说，几何平均滤波器的平滑效果 可与算术平均滤波器相媲美，但它会损失较少的图像细节。
    注意：如果图像的动态范围很大，我们一般会做log运算，但是对数运算后一般不使用几何平均滤波器。
    :param input_image:
    :param size:
    :return:
    """
    img = np.array(input_image, dtype=float)
    row, col = img.shape
    step = (size - 1) // 2
    if step * 2 + 1 >= row or step * 2 + 1 >= col:
        print("The parameter size is to large.")
    img_pad = np.pad(img, [step, step], 'constant', constant_values=[1] * 2)
    img_out = np.array(np.zeros((row, col)))
    for i in range(step, row):
        for j in range(step, col):
            temp = img_pad[i - step:i + step + 1, j - step:j + step + 1]
            temp = np.prod(temp)
            temp = np.power(temp, 1 / (size ** 2))
            img_out[i - step, j - step] = temp
    img_out = normalize(img_out)
    return img_out


def harmonic_mean(input_image, size):
    """
    它对盐噪声的效果很好，但对椒噪声则效果不好。它对其他类型的噪声如高斯噪声也有很好的效果。
    :param input_image:
    :param size:
    :return:
    """
    img = np.array(input_image, dtype=int)
    row, col = img.shape
    step = (size - 1) // 2
    if step * 2 + 1 >= row or step * 2 + 1 >= col:
        print("The parameter size is to large.")

    img_pad = np.pad(img, [step, step], 'constant', constant_values=[1] * 2)
    img_out = np.array(np.zeros((row, col)))
    for i in range(step, row):
        for j in range(step, col):
            temp = np.array(img_pad[i - step:i + step + 1, j - step:j + step + 1], dtype=float)
            temp = np.reciprocal(temp)
            temp = (size ** 2) / np.sum(temp)
            img_out[i - step, j - step] = temp
    img_out = normalize(img_out)
    return img_out


def contraharmonic_mean(input_image, q, size):
    """
    它非常适合于减少椒盐噪声的影响。Q>0处理胡椒噪声，Q<0处理盐噪声。
    缺点：不能同时处理椒和盐的噪声；

    :param input_image:
    :param q:Q>0 会导致黑色区域缩小，白色区域放大；Q<0 会导致白色区域缩小，黑色区域放大。
    :param size:
    :return:
    """
    global q2_array, q_array
    img = np.array(input_image, dtype=float)
    row, col = img.shape
    step = (size - 1) // 2
    if step * 2 + 1 >= row or step * 2 + 1 >= col:
        print("The parameter size is to large.")

    img_pad = np.pad(img, [step, step], 'constant', constant_values=[1] * 2)
    img_out = np.array(np.zeros((row, col)))

    if q > 0:
        q_array = np.array(np.maximum(np.zeros((size, size)), q), dtype=float)
        q2_array = np.array(np.maximum(np.zeros((size, size)), q + 1), dtype=float)
    elif q < 0:
        q_array = np.array(np.minimum(np.zeros((size, size)), q), dtype=float)
        q2_array = np.array(np.minimum(np.zeros((size, size)), q + 1), dtype=float)

    for i in range(step, row):
        for j in range(step, col):
            temp = np.array(img_pad[i - step:i + step + 1, j - step:j + step + 1], dtype=float)
            a = np.sum(np.power(temp, q2_array))
            b = np.sum(np.power(temp, q_array))
            img_out[i - step, j - step] = a / b

    img_out = normalize(img_out)
    return img_out


if __name__ == '__main__':
    try:
        pass
    except KeyboardInterrupt:
        pass

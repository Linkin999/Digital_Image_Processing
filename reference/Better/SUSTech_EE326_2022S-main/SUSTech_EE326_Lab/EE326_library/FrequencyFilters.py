#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :FrequencyFilters.py
# @Time      :2022-04-23 18:15
# @Author    :钟新宇
import numpy as np


def ideal_lowpass_mask(a, b, d0):
    x = np.array(np.linspace(0, a - 1, a) - a / 2)
    y = np.array(np.linspace(0, b - 1, b) - b / 2)
    mask = np.zeros((a, b))
    for i in range(a):
        for j in range(b):
            d = np.sqrt(x[i] ** 2 + y[j] ** 2)
            if d <= d0:
                mask[i, j] = 1
    return mask


def gaussian_lowpass_mask(a, b, sigma):
    x, y = np.meshgrid(np.linspace(0, a - 1, a), np.linspace(0, b - 1, b))
    x = x - a / 2
    y = y - b / 2
    mask = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    return mask


def butterworth_lowpass_mask(a, b, center, n, sigma):
    cx, cy = center
    x, y = np.meshgrid(np.linspace(0, a - 1, a), np.linspace(0, b - 1, b))
    x = x - cx
    y = y - cy
    d = np.sqrt(x * x + y * y)
    mask = 1 / ((1 + (d / sigma)) ** (2 * n))
    return mask


if __name__ == '__main__':
    try:
        pass
    except KeyboardInterrupt:
        pass

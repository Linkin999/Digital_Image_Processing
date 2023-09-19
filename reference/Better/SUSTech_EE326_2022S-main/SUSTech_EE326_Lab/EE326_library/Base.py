#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :Base.py
# @Time      :2022-04-23 18:00
# @Author    :钟新宇
import numpy as np
from matplotlib import pyplot as plt


def to_center(img, dtype=float):
    img = np.array(img, dtype=dtype)
    row, col = img.shape
    u, v = np.meshgrid(np.linspace(1, row, row), np.linspace(1, col, col))
    array = np.full((row, col), -1)
    temp = np.power(array, u + v)
    img_out = np.multiply(temp, img)
    return img_out


def normalize(input_array, mag_max=255, mag_min=0, dtype=float):
    input_array = np.array(input_array, dtype=dtype)
    zero_array = np.array(np.zeros(input_array.shape), dtype=dtype)
    max_array = np.array(np.full(input_array.shape, mag_max), dtype=dtype)
    zmax = np.max(input_array)
    zmin = np.min(input_array)
    output_array = input_array
    if zmax == zmin:
        if mag_min <= zmax <= mag_max:
            output_array = np.array(input_array, dtype=int)
        elif zmax < mag_min:
            output_array = np.array(zero_array, dtype=int)
        elif zmax > mag_max:
            output_array = np.array(max_array, dtype=int)
    else:
        temp_array = mag_max * (input_array - zmin) / (zmax - zmin)
        output_array = np.array(temp_array, dtype=int)
    return output_array


def plot(img, title, path):
    plt.figure()
    plt.title(title)
    plt.imshow(img, cmap='gray')
    plt.tight_layout()
    plt.savefig(path)
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    try:
        pass
    except KeyboardInterrupt:
        pass

#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :Degradations.py
# @Time      :2022-04-23 16:27
# @Author    :钟新宇
import numpy as np
from numpy import pi, sin
from numpy.fft import fft2, fftshift, ifft2, ifftshift

from EE326_library.Base import normalize, to_center


def full_inverse(input_image, h):
    """
    逆滤波：假设没有噪声，只考虑退化函数
    :param input_image:
    :param h:
    :return:
    """
    img = np.array(input_image, dtype=float)
    img_fft = fftshift(fft2(img))
    img_out_fft = img_fft / h
    img_out = np.real(ifft2(ifftshift(img_out_fft)))
    img_out = normalize(img_out)
    return img_out


def limit_inverse(input_image, h, radius):
    img = np.array(input_image, dtype=float)
    img_fft = fftshift(fft2(img))
    row, col = img_fft.shape
    img_out_fft = np.array(np.zeros(img_fft.shape), dtype=complex)
    for i in range(1, row + 1):
        for j in range(1, col + 1):
            if ((i - row / 2) ** 2 + (j - col / 2) ** 2) < radius ** 2:
                img_out_fft[i - 1, j - 1] = img_fft[i - 1, j - 1] / h[i - 1, j - 1]
    img_out = np.real(ifft2(ifftshift(img_out_fft)))
    img_out = normalize(img_out)
    return img_out


def wiener(input_image, h, k2):
    """
    维纳滤波：最小均方误差
    :param input_image:
    :param h:
    :param k2:
    :return:
    """
    img = np.array(input_image, dtype=float)
    img_fft = fftshift(fft2(img))
    h_conj = np.conjugate(h)
    h2 = np.multiply(h_conj, h)
    img_out_fft = img_fft * h2 / (h * (h2 + k2))
    img_out = np.real(ifft2(ifftshift(img_out_fft)))
    img_out = normalize(img_out)
    return img_out


def turbulence(input_image, k):
    """
    大气湍流的退化函数
    :param input_image:
    :param k:
    :return:
    """
    img = np.array(input_image, dtype=float)
    img_fft = fftshift(fft2(img))
    row, col = img_fft.shape
    u, v = np.meshgrid(np.linspace(0, row - 1, row), np.linspace(0, col - 1, col))
    u = u - row / 2
    v = v - col / 2
    d = np.power(u, 2) + np.power(v, 2)
    h = np.exp(-(k * (np.power(d, 5 / 6))))
    return h


def motion_blur(input_image, a, b, T):
    """
    相机运动模糊的退化函数
    :param input_image:
    :param a:
    :param b:
    :param T:
    :return:
    """
    img = np.array(input_image, dtype=float)
    # img = to_center(img)
    img_fft = fftshift(fft2(img))
    row, col = img_fft.shape
    u, v = np.meshgrid(np.linspace(1, row, row), np.linspace(1, col, col))
    d = pi * (u * a + v * b)
    e = np.exp(-1j * d)
    t = np.full([row, col], T)
    h = t * sin(d) * e / d
    return h


if __name__ == '__main__':
    try:
        pass
    except KeyboardInterrupt:
        pass

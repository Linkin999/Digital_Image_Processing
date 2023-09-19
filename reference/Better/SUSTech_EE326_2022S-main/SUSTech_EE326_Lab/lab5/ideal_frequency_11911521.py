#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :ideal_frequency_11911521.py
# @Time      :2022-04-02 1:24
# @Author    :钟新宇
import numpy as np
import cv2
from matplotlib import pyplot as plt


def range_normalize(input_array):
    output_array = np.array(input_array, dtype=int)
    # output_array = np.array(255 * np.divide(input_array, max(input_array.flat)), dtype=int)
    for i in range(output_array.shape[0]):
        for j in range(output_array.shape[1]):
            if output_array[i, j] < 0:
                output_array[i, j] = 0
            elif output_array[i, j] > 255:
                output_array[i, j] = 255
    return output_array


def shift_matrix(input_image):
    row, col = input_image.shape
    for i in range(row):
        for j in range(col):
            input_image[i, j] = input_image[i, j] * (-1) ** (i + j)
    return input_image


def ideal_mask(a, b, d0):
    x = np.array(np.linspace(0, a - 1, a) - a / 2)
    y = np.array(np.linspace(0, b - 1, b) - b / 2)

    mask = np.zeros((a, b))
    for i in range(a):
        for j in range(b):
            d = np.sqrt(x[i] ** 2 + y[j] ** 2)
            if d <= d0:
                mask[i, j] = 1
    return mask


def ideal_frequency(input_image, d0):
    row, col = input_image.shape

    img_fft = np.fft.fft2(input_image)
    img_fft_shift = np.fft.fftshift(img_fft)

    kernel_lpf_fft = ideal_mask(row, col, d0)
    kernel_hpf_fft = np.ones((row, col)) - kernel_lpf_fft

    img_lpf_filtered = np.multiply(img_fft_shift, kernel_lpf_fft)
    img_lpf_filtered = np.fft.fftshift(img_lpf_filtered)
    img_lpf_filtered = np.fft.ifft2(img_lpf_filtered)
    img_lpf_filtered = np.real(img_lpf_filtered)
    img_lpf_filtered = range_normalize(img_lpf_filtered)

    img_hpf_filtered = np.multiply(img_fft_shift, kernel_hpf_fft)
    img_hpf_filtered = np.fft.fftshift(img_hpf_filtered)
    img_hpf_filtered = np.fft.ifft2(img_hpf_filtered)
    img_hpf_filtered = np.real(img_hpf_filtered)
    img_hpf_filtered = range_normalize(img_hpf_filtered)

    img_fft_shift = range_normalize(img_fft_shift)
    return img_fft_shift, kernel_lpf_fft, kernel_hpf_fft, img_lpf_filtered, img_hpf_filtered


if __name__ == '__main__':
    try:
        img_init = cv2.imread("./Q5_2.tif", cv2.IMREAD_GRAYSCALE)
        img_init = np.array(img_init, dtype=int)

        d0 = [10, 30, 60, 160, 460]

        for i in d0:
            [img_fft_shift, kernel_lpf_fft, kernel_hpf_fft, img_lpf_filtered, img_hpf_filtered] = ideal_frequency(
                img_init, d0=i)
            plt.figure()
            plt.title("img_fft_shift+D0=%d" % i)
            plt.imshow(img_fft_shift, cmap='gray')
            plt.show()

            plt.figure()
            plt.title("kernel_lpf_fft+D0=%d" % i)
            plt.imshow(kernel_lpf_fft, cmap='gray')
            plt.show()

            plt.figure()
            plt.title("img_lpf_filtered+D0=%d" % i)
            plt.imshow(img_lpf_filtered, cmap='gray')
            plt.show()

            plt.figure()
            plt.title("kernel_hpf_fft+D0=%d" % i)
            plt.imshow(kernel_hpf_fft, cmap='gray')
            plt.show()

            plt.figure()
            plt.title("img_hpf_filtered+D0=%d" % i)
            plt.imshow(img_hpf_filtered, cmap='gray')
            plt.show()
    except KeyboardInterrupt:
        pass

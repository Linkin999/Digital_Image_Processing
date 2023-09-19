#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :gaussian_frequency_11911521.py
# @Time      :2022-04-02 1:23
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


def gaussian_mask(a, b, sigma):
    x, y = np.meshgrid(np.linspace(0, a - 1, a), np.linspace(0, b - 1, b))
    x = x - a / 2
    y = y - b / 2
    mask = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    return mask


def gaussian_frequency(input_image, sigma):
    row, col = input_image.shape

    img_fft = np.fft.fft2(input_image)
    img_fft_shift = np.fft.fftshift(img_fft)

    kernel_lpf_fft = gaussian_mask(row, col, sigma)
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
        img_init = cv2.imread("./Q5_1.tif", cv2.IMREAD_GRAYSCALE)
        img_init = np.array(img_init, dtype=int)

        sigma = [30, 60, 160]

        for i in sigma:
            [img_fft_shift, kernel_lpf_fft, kernel_hpf_fft, img_lpf_filtered, img_hpf_filtered] = gaussian_frequency(
                img_init, sigma=i)
            plt.figure()
            plt.title("img_fft_shift+sigma=%d" % i)
            plt.imshow(img_fft_shift, cmap='gray')
            plt.show()

            plt.figure()
            plt.title("kernel_lpf_fft+sigma=%d" % i)
            plt.imshow(kernel_lpf_fft, cmap='gray')
            plt.show()

            plt.figure()
            plt.title("img_lpf_filtered+sigma=%d" % i)
            plt.imshow(img_lpf_filtered, cmap='gray')
            plt.show()

            plt.figure()
            plt.title("kernel_hpf_fft+sigma=%d" % i)
            plt.imshow(kernel_hpf_fft, cmap='gray')
            plt.show()

            plt.figure()
            plt.title("img_hpf_filtered+sigma=%d" % i)
            plt.imshow(img_hpf_filtered, cmap='gray')
            plt.show()
    except KeyboardInterrupt:
        pass

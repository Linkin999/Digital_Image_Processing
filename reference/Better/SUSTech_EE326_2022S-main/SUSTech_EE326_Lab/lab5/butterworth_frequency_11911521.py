#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :butterworth_frequency_11911521.py
# @Time      :2022-04-02 1:25
# @Author    :钟新宇
import numpy as np
import cv2
from matplotlib import pyplot as plt

centers = [[109, 87], [109, 170], [115, 330], [115, 412], [227, 405], [227, 325], [223, 162], [223, 79]]


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


def butterworth_mask(a, b, center, n, sigma):
    cx, cy = center
    x, y = np.meshgrid(np.linspace(0, a - 1, a), np.linspace(0, b - 1, b))
    x = x - cx
    y = y - cy
    d = np.sqrt(x * x + y * y)
    mask = 1 / ((1 + (d / sigma)) ** (2 * n))
    return mask


def butterworth_frequency(input_image, centers):
    row, col = input_image.shape
    img_pad = np.pad(input_image, ((0, row), (0, col)))

    p, q = img_pad.shape
    img_fft = np.fft.fft2(img_pad)
    img_fft_shift = np.fft.fftshift(img_fft)

    kernel_lpf_fft = np.zeros((p, q))
    for c in centers:
        kernel_lpf_fft += butterworth_mask(q, p, c, 4, 100)
    kernel_hpf_fft = np.ones((p, q)) - kernel_lpf_fft

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

        img = cv2.imread("./Q5_3.tif", cv2.IMREAD_GRAYSCALE)
        img_init = np.array(img, dtype=int)

        [img_fft_shift, kernel_lpf_fft, kernel_hpf_fft, img_lpf_filtered, img_hpf_filtered] = butterworth_frequency(
            img_init, centers=centers)

        plt.figure()
        plt.title("img_fft_shift")
        plt.imshow(img_fft_shift, cmap='gray')
        plt.show()

        plt.figure()
        plt.title("kernel_lpf_fft")
        plt.imshow(kernel_lpf_fft, cmap='gray')
        plt.show()

        plt.figure()
        plt.subplot(121)
        plt.title("img_init")
        plt.imshow(img, cmap='gray')
        plt.subplot(122)
        plt.title("img_lpf_filtered")
        plt.imshow(img_lpf_filtered, cmap='gray')
        plt.show()

        plt.figure()
        plt.title("kernel_hpf_fft")
        plt.imshow(kernel_hpf_fft, cmap='gray')
        plt.show()

        plt.figure()
        plt.subplot(121)
        plt.title("img_init")
        plt.imshow(img, cmap='gray')
        plt.subplot(122)
        plt.title("img_hpf_filtered")
        plt.imshow(img_hpf_filtered, cmap='gray')
        plt.show()
    except KeyboardInterrupt:
        pass

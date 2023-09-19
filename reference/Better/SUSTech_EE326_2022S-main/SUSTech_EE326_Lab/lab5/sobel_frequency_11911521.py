#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :sobel_frequency_11911521.py
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


def sobel_spatial(input_image):
    row, col = input_image.shape
    kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=int)

    img_pad = np.pad(input_image, 1)

    img_out = np.zeros((row, col), dtype=int)
    for i in range(row):
        for j in range(col):
            temp = np.sum(img_pad[i:i + 3, j:j + 3] * kernel)
            img_out[i, j] = temp
    img_out = range_normalize(img_out)
    return img_out


def sobel_frequency(input_image):
    row, col = input_image.shape

    # sobel kernel
    kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=int)
    kernel_pad = np.pad(kernel, ((0, 2 * row - 3), (0, 2 * col - 3)))
    kernel_fft = np.fft.fft2(kernel_pad)
    kernel_fft_shift = np.fft.fftshift(kernel_fft)

    # zero pad

    img_pad = np.pad(input_image, ((0, row), (0, col)))
    # filtered image without shift
    img_fft = np.fft.fft2(img_pad)
    img_filtered = np.multiply(img_fft, kernel_fft)
    # img_filtered = np.fft.fftshift(img_filtered)
    img_filtered = np.fft.ifft2(img_filtered)
    img_filtered = np.real(img_filtered)
    img_filtered = range_normalize(img_filtered)

    # filtered image with shift
    img_fft_shift = np.fft.fftshift(img_fft)
    # img_pad = np.pad(input_image, ((0, row), (0, col)))
    # img_fft_shift = np.fft.fft2(img_pad)
    img_filtered_shift = np.multiply(img_fft_shift, kernel_fft_shift)
    # img_filtered_shift = np.fft.fftshift(img_filtered_shift)
    img_filtered_shift = np.fft.ifft2(img_filtered_shift)
    img_filtered_shift = np.real(img_filtered_shift)
    img_filtered_shift = range_normalize(img_filtered_shift)

    return img_pad, kernel_pad, kernel_fft, img_fft_shift, img_filtered_shift, img_fft, img_filtered


if __name__ == '__main__':
    try:
        img_init = cv2.imread("Q5_1.png", cv2.IMREAD_GRAYSCALE)
        img_init = np.array(img_init)

        [img_pad_out, kernel_pad_out, kernel_fft_out, img_fft_shift_out, img_filtered_shift_out, img_fft_out, img_filtered_out] = sobel_frequency(img_init)
        img_spatial = sobel_spatial(img_init)

        plt.figure()
        plt.title("img_spatial")
        plt.imshow(img_spatial, cmap='gray')
        plt.show()

        plt.figure()
        plt.title("img_init")
        plt.imshow(img_init, cmap='gray')
        plt.show()

        plt.figure()
        plt.title("img_pad_out")
        plt.imshow(img_pad_out, cmap='gray')
        plt.show()

        # plt.figure()
        # plt.title("img_fft_shift_out")
        # plt.imshow(img_fft_shift_out, cmap='gray')
        # plt.show()

        plt.figure()
        plt.title("img_filtered_shift_out")
        plt.imshow(img_filtered_shift_out, cmap='gray')
        plt.show()

        # plt.figure()
        # plt.title("img_fft_out")
        # plt.imshow(img_fft_out, cmap='gray')
        # plt.show()

        plt.figure()
        plt.title("img_filtered_out")
        plt.imshow(img_filtered_out, cmap='gray')
        plt.show()
    except KeyboardInterrupt:
        pass

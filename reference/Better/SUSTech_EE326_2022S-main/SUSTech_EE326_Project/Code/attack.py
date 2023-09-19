#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :attack.py
# @Time      :2022-05-24 21:01
# @Author    :钟新宇

import numpy as np
from matplotlib import pyplot as plt
from Base import showimg, saveimg, img2rgb, rgb2img
from lib.MeanFilters import arithmetic_mean
from lib.OrderStatisticFilters import median_filter
from wm_lsb import decode
import cv2
import random


def psnr(src, rec):
    # read image
    img_src, img_rec = np.asarray(src, dtype=int), np.asarray(rec, dtype=int)
    # get size
    m, n, _ = img_src.shape
    # get rgb component
    src_r, src_g, src_b = img2rgb(img_src)
    rec_r, rec_g, rec_b = img2rgb(img_rec)
    # calculate error
    error_r, error_g, error_b = np.abs(np.subtract(src_r, rec_r)), np.abs(np.subtract(src_g, rec_g)), np.abs(np.subtract(src_b, rec_b))
    # calculate sum
    sum_r, sum_g, sum_b = np.sum(error_r), np.sum(error_g), np.sum(error_b)
    # calculate mse
    mse_r, mse_g, mse_b = np.multiply(sum_r, 1 / (m * n)), np.multiply(sum_g, 1 / (m * n)), np.multiply(sum_b, 1 / (m * n))
    # find maximum
    max_r, max_g, max_b = np.max(src_r), np.max(src_g), np.max(src_b)
    # calculate snr
    psnr_r = 10 * np.log10(max_r ** 2 / mse_r) if mse_r != 0 else -np.inf
    psnr_g = 10 * np.log10(max_g ** 2 / mse_g) if mse_g != 0 else -np.inf
    psnr_b = 10 * np.log10(max_b ** 2 / mse_b) if mse_b != 0 else -np.inf
    snr = np.asarray([psnr_r, psnr_g, psnr_b], dtype=float)
    print(snr)
    return snr


def nc(src, rec):
    # read image
    mark_src = np.asarray(src, dtype=int)
    mark_rec = np.asarray(rec, dtype=int)
    # get size
    m, n, _ = mark_src.shape
    # get rgb component
    src_r, src_g, src_b = img2rgb(mark_src)
    rec_r, rec_g, rec_b = img2rgb(mark_rec)
    # calculate sum1
    sum1_r = np.sum(np.multiply(src_r, rec_r))
    sum1_g = np.sum(np.multiply(src_g, rec_g))
    sum1_b = np.sum(np.multiply(src_b, rec_b))
    # calculate sum2 (src)
    sum2_r = np.sqrt(np.sum(np.power(src_r, 2)))
    sum2_g = np.sqrt(np.sum(np.power(src_g, 2)))
    sum2_b = np.sqrt(np.sum(np.power(src_b, 2)))
    # calculate sum3 (rec)
    sum3_r = np.sqrt(np.sum(np.power(rec_r, 2)))
    sum3_g = np.sqrt(np.sum(np.power(rec_g, 2)))
    sum3_b = np.sqrt(np.sum(np.power(rec_b, 2)))
    # calculate nc
    nc_r = sum1_r / (sum2_r * sum3_r)
    nc_g = sum1_g / (sum2_g * sum3_g)
    nc_b = sum1_b / (sum2_b * sum3_b)
    nc_out = np.asarray([nc_r, nc_g, nc_b], dtype=float)
    print(nc_out)
    return nc_out


def noise_sp(img, prop):
    img_noise = np.asarray(img, dtype=int)
    m, n, _ = img_noise.shape
    num = int(m * n * prop)
    for i in range(num):
        w = random.randint(0, n - 1)
        h = random.randint(0, m - 1)
        if random.randint(0, 1) == 0:
            img_noise[h, w] = 0
        else:
            img_noise[h, w] = 255
    showimg(img_noise, title="img_noise_sp")
    saveimg(img_noise, path="./img/img_noise_sp.bmp")
    return img_noise


def noise_gaussian(img, mean, sigma):
    img_noise = np.asarray(img, dtype=int)
    noise = np.random.normal(mean, sigma, img_noise.shape)
    img_noise = img_noise + noise
    img_noise = np.clip(img_noise, 0, 255)
    showimg(img_noise, title="img_noise_gaussian")
    saveimg(img_noise, path="./img/img_noise_gaussian.bmp")
    return img_noise


def noise_random(img, noise_num):
    img_noise = np.asarray(img, dtype=int)
    rows, cols, _ = img_noise.shape
    for i in range(noise_num):
        x = np.random.randint(0, rows)  # 随机生成指定范围的整数
        y = np.random.randint(0, cols)
        img_noise[x, y, :] = 255
    showimg(img_noise, title="img_noise_random")
    saveimg(img_noise, path="./img/img_noise_random.bmp")
    return img_noise


def hist_equ(img):
    img = np.asarray(img, dtype=int)
    m, n = img.shape
    L = 256
    bins = range(L + 1)
    hist_in, _ = np.histogram(img.flat, bins=bins, density=True)

    s = np.asarray(np.zeros(256))
    for i in range(L):
        s[i] = (L - 1) * sum(hist_in[:i + 1])

    img_out = np.asarray(np.zeros((m, n)), dtype=int)
    for i in range(m):
        for j in range(n):
            img_out[i][j] = s[img[i][j]]

    return img_out


def histogram_equal(img):
    img_equal = np.asarray(img, dtype=int)
    m, n, _ = img_equal.shape
    r, g, b = img2rgb(img_equal)
    r = np.clip(hist_equ(r), 0, 255)
    g = np.clip(hist_equ(g), 0, 255)
    b = np.clip(hist_equ(b), 0, 255)
    img_equal = rgb2img((m, n), r, g, b)
    showimg(img_equal, title="img_histogram_equal")
    saveimg(img_equal, path="./img/img_histogram_equal.bmp")
    return img_equal


def mean(img):
    img_filter = np.asarray(img, dtype=int)
    m, n, _ = img_filter.shape
    size = 3
    r, g, b = img2rgb(img_filter)
    r = arithmetic_mean(r, size=size)
    g = arithmetic_mean(g, size=size)
    b = arithmetic_mean(b, size=size)
    img_filter = rgb2img((m, n), r, g, b)
    showimg(img_filter, title="img_filter_mean")
    saveimg(img_filter, path="./img/img_filter_mean.bmp")
    return img_filter


def median(img):
    img_filter = np.asarray(img, dtype=int)
    m, n, _ = img_filter.shape
    size = 3
    r, g, b = img2rgb(img_filter)
    r = median_filter(r, size=size)
    g = median_filter(g, size=size)
    b = median_filter(b, size=size)
    img_filter = rgb2img((m, n), r, g, b)
    showimg(img_filter, title="img_filter_median")
    saveimg(img_filter, path="./img/img_filter_median.bmp")
    return img_filter


if __name__ == '__main__':
    try:
        """
        src
        "./img/img_src.bmp",  # img_src_path
        "./img/img_mark.bmp"  # img_mark_path
        "./img/mark_src.bmp",  # mark_src_path
        "./img/mark_enc.bmp",  # mark_enc_path
        "./img/mark_rec.bmp"  # mark_rec_path
        
        noise
        "./img/img_noise_sp.bmp"
        "./img/mark_noise_sp.bmp"
        "./img/img_noise_gaussian.bmp"
        "./img/mark_noise_gaussian.bmp"
        "./img/img_noise_random.bmp"
        "./img/mark_noise_random.bmp"
        
        histogram equalization
        "./img/img_histogram_equal.bmp"
        "./img/mark_histogram_equal.bmp"
        
        filter
        "./img/img_filter_mean.bmp"
        "./img/mark_filter_mean.bmp"
        "./img/img_filter_median.bmp"
        "./img/mark_filter_median.bmp"
        """

        mark_size = np.asarray([256, 256], dtype=int)

        img_src = np.asarray(plt.imread("./img/img_src.bmp"), dtype=int)
        img_rec = np.asarray(plt.imread("./img/img_mark.bmp"), dtype=int)
        mark_src = np.asarray(plt.imread("./img/mark_src.bmp"), dtype=int)
        mark_rec = np.asarray(plt.imread("./img/mark_rec.bmp"), dtype=int)

        """noise attack"""
        # img_noise_sp = noise_sp(img_rec, prop=0.1)
        # img_noise_gaussian = noise_gaussian(img_rec, mean=10, sigma=255)
        # img_noise_random = noise_random(img_rec, noise_num=100)
        #
        # decode(path=["./img/img_noise_sp.bmp", "./img/mark_noise_sp.bmp"], size_mark=mark_size, key=11)
        # decode(path=["./img/img_noise_gaussian.bmp", "./img/mark_noise_gaussian.bmp"], size_mark=mark_size, key=11)
        # decode(path=["./img/img_noise_random.bmp", "./img/mark_noise_random.bmp"], size_mark=mark_size, key=11)

        """histogram equalization"""
        # img_equal = histogram_equal(img_rec)
        # decode(path=["./img/img_histogram_equal.bmp", "./img/mark_histogram_equal.bmp"], size_mark=mark_size, key=11)

        """filter"""
        # img_filter_mean = mean(img_rec)
        # decode(path=["./img/img_filter_mean.bmp", "./img/mark_filter_mean.bmp"], size_mark=mark_size, key=11)
        # img_filter_median = median(img_rec)
        # decode(path=["./img/img_filter_median.bmp", "./img/mark_filter_median.bmp"], size_mark=mark_size, key=11)

        """psnr and nc"""
        # psnr_src = psnr(src=img_src, rec=img_src)
        psnr_1 = psnr(src=img_src, rec=img_rec)
        nc_1 = nc(mark_src, mark_rec)
    except KeyboardInterrupt:
        pass

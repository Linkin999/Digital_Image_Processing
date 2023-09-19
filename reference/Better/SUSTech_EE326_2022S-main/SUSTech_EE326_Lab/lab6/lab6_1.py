#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :lab6_1.py
# @Time      :2022-04-23 18:33
# @Author    :钟新宇
import cv2
import numpy as np
from matplotlib import pyplot as plt

from EE326_library.Base import plot
from EE326_library.AdaptiveFilters import adaptive_arithmetic_mean, adaptive_median_filter
from EE326_library.MeanFilters import arithmetic_mean, geometric_mean, harmonic_mean, contraharmonic_mean
from EE326_library.OrderStatisticFilters import median_filter, max_filter, min_filter, midpoint_filter, \
    alpha_trimmed_mean


def lab6_1(img, size, path, q, noise_var, d, smax):
    img = np.maximum(img, 1)
    img_arithmetic_mean = arithmetic_mean(img, size=size)
    img_geometric_mean = geometric_mean(img, size=size)
    img_harmonic_mean = harmonic_mean(img, size=size)
    img_contraharmonic_mean_1 = contraharmonic_mean(img, size=size, q=q)
    img_contraharmonic_mean_2 = contraharmonic_mean(img, size=size, q=-q)
    img_median_filter = median_filter(img, size=size)
    img_max_filter = max_filter(img, size=size)
    img_min_filter = min_filter(img, size=size)
    img_midpoint_filter = midpoint_filter(img, size=size)
    img_alpha_trimmed_mean = alpha_trimmed_mean(img, d=d, size=size)
    img_adaptive_arithmetic_mean = adaptive_arithmetic_mean(img, size=size, noise_var=noise_var)
    img_adaptive_median_filter = adaptive_median_filter(img, smax=smax, smin=1)

    plot(img=img, title="img", path="./img_result/" + path + "/img.png")
    plot(img=img_arithmetic_mean, title="img_arithmetic_mean", path="./img_result/" + path + "/img_arithmetic_mean.png")
    plot(img=img_geometric_mean, title="img_geometric_mean", path="./img_result/" + path + "/img_geometric_mean.png")
    plot(img=img_harmonic_mean, title="img_harmonic_mean", path="./img_result/" + path + "/img_harmonic_mean.png")
    plot(img=img_contraharmonic_mean_1, title="img_contraharmonic_mean_1" + "\n" + "q=%f" % q,
         path="./img_result/" + path + "/img_contraharmonic_mean_1.png")
    plot(img=img_contraharmonic_mean_2, title="img_contraharmonic_mean_1" + "\n" + "q=%f" % q,
         path="./img_result/" + path + "/img_contraharmonic_mean_2.png")
    plot(img=img_median_filter, title="img_median_filter", path="./img_result/" + path + "/img_median_filter.png")
    plot(img=img_max_filter, title="img_max_filter", path="./img_result/" + path + "/img_max_filter.png")
    plot(img=img_min_filter, title="img_min_filter", path="./img_result/" + path + "/img_min_filter.png")
    plot(img=img_midpoint_filter, title="img_midpoint_filter", path="./img_result/" + path + "/img_midpoint_filter.png")
    plot(img=img_alpha_trimmed_mean, title="img_alpha_trimmed_mean" + "\n" + "d=%d" % d,
         path="./img_result/" + path + "/img_alpha_trimmed_mean.png")
    plot(img=img_adaptive_arithmetic_mean, title="img_adaptive_arithmetic_mean" + "\n" + "noise var=%f" % noise_var,
         path="./img_result/" + path + "/img_adaptive_arithmetic_mean.png")
    plot(img=img_adaptive_median_filter, title="img_adaptive_median_filter""\n" + "smax=%d" % smax,
         path="./img_result/" + path + "/img_adaptive_median_filter.png")
    plot(img=img_midpoint_filter, title="img_midpoint_filter", path="./img_result/" + path + "/img_midpoint_filter.png")


def lab6_1_1():
    """
    pepper noise; FIGURE 5.8; Page = 325
    :return:
    """
    img = np.asarray(cv2.imread("./img_source/Q6_1_1.tiff", cv2.IMREAD_GRAYSCALE), dtype=int)
    lab6_1(img, size=3, path="lab6_1_1", q=1.5, noise_var=0.1, d=2, smax=7)


def lab6_1_2():
    """
    salt noise; FIGURE 5.8; Page = 325
    :return:
    """
    img = np.asarray(cv2.imread("./img_source/Q6_1_2.tiff", cv2.IMREAD_GRAYSCALE), dtype=int)
    lab6_1(img, size=3, path="lab6_1_2", q=1.5, noise_var=0.1, d=2, smax=7)


def lab6_1_3():
    """
    pepper and salt noise; FIGURE 5.12; Page = 329
    :return:
    """
    img = np.asarray(cv2.imread("./img_source/Q6_1_3.tiff", cv2.IMREAD_GRAYSCALE), dtype=int)
    lab6_1(img, size=5, path="lab6_1_3", q=1.5, noise_var=0.25, d=2, smax=7)


def lab6_1_4():
    """
    uniform noise and pepper noise and salt noise;
    FIGURE 5.14; Page = 334
    :return:
    """
    img = np.asarray(cv2.imread("./img_source/Q6_1_4.tiff", cv2.IMREAD_GRAYSCALE), dtype=int)
    lab6_1(img, size=7, path="lab6_1_4", q=1.5, noise_var=0.25, d=2, smax=7)


if __name__ == '__main__':
    try:
        lab6_1_1()
        lab6_1_2()
        lab6_1_3()
        lab6_1_4()
    except KeyboardInterrupt:
        pass

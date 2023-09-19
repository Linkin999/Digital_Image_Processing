#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :lab6_3.py
# @Time      :2022-04-23 18:33
# @Author    :钟新宇
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from EE326_library.Base import plot
from EE326_library.Degradations import motion_blur, full_inverse, limit_inverse, wiener
from lab6.lab6_1 import lab6_1


def lab6_motion_blur(img, path, a, b, T, mode, radius=70, k2=100):
    h = motion_blur(img, a=a, b=b, T=T)
    if mode == "full":
        img_motion_blur_full = full_inverse(img, h=h)
        plot(img=img_motion_blur_full, title="img_motion_blur_full",
             path="./img_result/" + path + "/img_motion_blur_full.png")
    elif mode == "limit":
        img_motion_blur_limit = limit_inverse(img, h=h, radius=radius)
        plot(img=img_motion_blur_limit, title="img_motion_blur_limit",
             path="./img_result/" + path + "/img_motion_blur_limit.png")
    elif mode == "wiener":
        img_motion_blur_wiener = wiener(img, h=h, k2=k2)
        plot(img=img_motion_blur_wiener, title="img_motion_blur_wiener",
             path="./img_result/" + path + "/img_motion_blur_wiener.png")


def lab6_3_1():
    """
    no noise
    :return:
    """
    img = np.asarray(cv2.imread("./img_source/Q6_3_1.tiff", cv2.IMREAD_GRAYSCALE), dtype=int)
    # img = plt.imread("./img_source/Q6_3_1.tiff")
    img = np.array(img, dtype=int)
    # plt.figure()
    # plt.imshow(img, cmap='gray')
    # plt.show()
    lab6_motion_blur(img=img, path="lab6_3_1", a=0.1, b=0.1, T=1, mode="full")
    lab6_motion_blur(img=img, path="lab6_3_1", a=0.1, b=0.1, T=1, mode="limit", radius=40)
    lab6_motion_blur(img=img, path="lab6_3_1", a=0.1, b=0.1, T=1, mode="wiener", k2=100)


def lab6_3_2():
    """
    uniform noise;
    :return:
    """
    img = np.asarray(cv2.imread("./img_source/Q6_3_2.tiff", cv2.IMREAD_GRAYSCALE), dtype=int)
    lab6_1(img, size=3, path="lab6_3_2", q=1.5, noise_var=0.1, d=2, smax=7)


def lab6_3_3():
    """
    pepper and salt noise;
    :return:
    """
    img = np.asarray(cv2.imread("./img_source/Q6_3_3.tiff", cv2.IMREAD_GRAYSCALE), dtype=int)
    lab6_1(img, size=5, path="lab6_3_3", q=1.5, noise_var=0.25, d=2, smax=7)


if __name__ == '__main__':
    try:
        # lab6_3_1()
        lab6_3_2()
        # lab6_3_3()

    except KeyboardInterrupt:
        pass

#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :Filter_Base.py
# @Time      :2022-05-23 16:30
# @Author    :钟新宇
import numpy as np
from matplotlib import pyplot as plt
import cv2


def dec2bit(num_dec):
    s = bin(num_dec)[2:]
    # print(len(s))
    while len(s) < 8:
        s = "0" + s  # 满足条件的高位补零
    num_bit = np.zeros(8, dtype=int)
    for o in range(8):
        num_bit[o] = s[o]
    return num_bit


def bit2dec(num_bit):
    arr = np.asarray(num_bit, dtype=str)
    s = "0b" + "".join(arr)
    num_dec = int(s, 2)
    return num_dec


def readimg(path):
    img_in = plt.imread(path)
    img_in = np.asarray(img_in, dtype=int)
    showimg(img_in, title="img_in")
    img_size = img_in.shape
    img_r = img_in[:, :, 0]
    img_g = img_in[:, :, 1]
    img_b = img_in[:, :, 2]

    img_r_show = np.zeros(img_size, dtype=int)
    img_r_show[:, :, 0] = img_r
    # showimg(img_r_show, title="img_r")
    img_g_show = np.zeros(img_size, dtype=int)
    img_g_show[:, :, 1] = img_g
    # showimg(img_g_show, title="img_g")
    img_b_show = np.zeros(img_size, dtype=int)
    img_b_show[:, :, 2] = img_b
    # showimg(img_b_show, title="img_b")
    return img_in, img_r, img_g, img_b


def showimg(img, title="Image"):
    plt.figure()
    plt.title(title)
    plt.imshow(img)
    plt.show()


def saveimg(img_rgb, path):
    path_str = np.str(path)
    bgr = rgb2bgr(img_rgb)
    cv2.imwrite(filename=path_str, img=bgr)


def img2rgb(rgb):
    img = np.asarray(rgb, dtype=int)
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]
    return r, g, b


def rgb2img(size, r, g, b):
    m, n = size
    rgb = np.zeros((m, n, 3), dtype=int)
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b
    return rgb


def rgb2bgr(rgb):
    rgb = np.asarray(rgb, dtype=int)
    bgr = np.zeros(rgb.shape, dtype=int)
    bgr[:, :, 0] = rgb[:, :, 2]
    bgr[:, :, 1] = rgb[:, :, 1]
    bgr[:, :, 2] = rgb[:, :, 0]
    return bgr


if __name__ == '__main__':
    try:
        pass
    except KeyboardInterrupt:
        pass

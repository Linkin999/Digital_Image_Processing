#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :lab6_2.py
# @Time      :2022-04-23 20:40
# @Author    :钟新宇
import cv2
import numpy as np

from EE326_library.Degradations import full_inverse, limit_inverse, wiener, turbulence
from EE326_library.Base import normalize, plot


def lab6_full_inverse(img, path, k):
    h_1 = turbulence(img, k=k[0])
    h_2 = turbulence(img, k=k[1])
    h_3 = turbulence(img, k=k[2])
    h_4 = turbulence(img, k=k[3])
    img_full_inverse_1 = full_inverse(img, h=h_1)
    img_full_inverse_2 = full_inverse(img, h=h_2)
    img_full_inverse_3 = full_inverse(img, h=h_3)
    img_full_inverse_4 = full_inverse(img, h=h_4)

    plot(img=img_full_inverse_1, title="k=%f" % k[0], path="./img_result/" + path + "/img_full_inverse_1.png")
    plot(img=img_full_inverse_2, title="k=%f" % k[1], path="./img_result/" + path + "/img_full_inverse_2.png")
    plot(img=img_full_inverse_3, title="k=%f" % k[2], path="./img_result/" + path + "/img_full_inverse_3.png")
    plot(img=img_full_inverse_4, title="k=%f" % k[3], path="./img_result/" + path + "/img_full_inverse_4.png")


def lab6_limit_inverse(img, path, k, radius):
    h = turbulence(img, k=k)
    img_limit_inverse_1 = limit_inverse(img, h=h, radius=radius[0])
    img_limit_inverse_2 = limit_inverse(img, h=h, radius=radius[1])
    img_limit_inverse_3 = limit_inverse(img, h=h, radius=radius[2])
    img_limit_inverse_4 = limit_inverse(img, h=h, radius=radius[3])

    plot(img=img_limit_inverse_1, title="radius=%f" % radius[0],
         path="./img_result/" + path + "/img_limit_inverse_1.png")
    plot(img=img_limit_inverse_2, title="radius=%f" % radius[1],
         path="./img_result/" + path + "/img_limit_inverse_2.png")
    plot(img=img_limit_inverse_3, title="radius=%f" % radius[2],
         path="./img_result/" + path + "/img_limit_inverse_3.png")
    plot(img=img_limit_inverse_4, title="radius=%f" % radius[3],
         path="./img_result/" + path + "/img_limit_inverse_4.png")


def lab6_wiener(img, path, k, k2):
    h = turbulence(img, k=k)
    img_wiener_1 = wiener(img, h=k, k2=k2[0])
    img_wiener_2 = wiener(img, h=h, k2=k2[1])
    img_wiener_3 = wiener(img, h=h, k2=k2[2])
    img_wiener_4 = wiener(img, h=h, k2=k2[3])
    img_wiener_5 = wiener(img, h=k, k2=k2[4])
    img_wiener_6 = wiener(img, h=h, k2=k2[5])
    img_wiener_7 = wiener(img, h=h, k2=k2[6])
    img_wiener_8 = wiener(img, h=h, k2=k2[7])
    img_wiener_9 = wiener(img, h=h, k2=k2[8])

    plot(img=img_wiener_1, title="k2=%f" % k2[0], path="./img_result/" + path + "/img_wiener_1.png")
    plot(img=img_wiener_2, title="k2=%f" % k2[1], path="./img_result/" + path + "/img_wiener_2.png")
    plot(img=img_wiener_3, title="k2=%f" % k2[2], path="./img_result/" + path + "/img_wiener_3.png")
    plot(img=img_wiener_4, title="k2=%f" % k2[3], path="./img_result/" + path + "/img_wiener_4.png")
    plot(img=img_wiener_5, title="k2=%f" % k2[4], path="./img_result/" + path + "/img_wiener_5.png")
    plot(img=img_wiener_6, title="k2=%f" % k2[5], path="./img_result/" + path + "/img_wiener_6.png")
    plot(img=img_wiener_7, title="k2=%f" % k2[6], path="./img_result/" + path + "/img_wiener_7.png")
    plot(img=img_wiener_8, title="k2=%f" % k2[7], path="./img_result/" + path + "/img_wiener_8.png")
    plot(img=img_wiener_9, title="k2=%f" % k2[8], path="./img_result/" + path + "/img_wiener_9.png")


def lab6_add_gaussian_noise(img, path, sigma):
    img = np.array(img, dtype=float)
    row, col = img.shape
    noise = np.random.normal(0, sigma, (row, col))
    img_out = img + noise
    img_out = normalize(img_out)

    plot(img=img, title="img", path="./img_result/" + path + "/img.png")
    plot(img=img_out, title="img_with_gaussian", path="./img_result/" + path + "/img_with_gaussian.png")
    return img_out


if __name__ == '__main__':
    try:
        img = cv2.imread("./img_source/Q6_2.tif", cv2.IMREAD_GRAYSCALE)
        path = "lab6_2"
        img = lab6_add_gaussian_noise(img=img, path=path, sigma=0.0065)
        lab6_full_inverse(img=img, path=path, k=np.array([2.5e-3, 1e-3, 2.5e-4, 1e-4], dtype=float))
        lab6_limit_inverse(img=img, path=path, k=2.5e-4, radius=np.array([40, 80, 120, 160], dtype=float))
        lab6_wiener(img=img, path=path, k=2.5e-4, k2=np.array([1e-20, 1e-15, 1e-10, 1e-5, 1, 1e5, 1e10, 1e15, 1e20], dtype=float))
        pass
    except KeyboardInterrupt:
        pass

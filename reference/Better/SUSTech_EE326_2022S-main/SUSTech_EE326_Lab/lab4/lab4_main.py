#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :lab4_main.py
# @Time      :2022-03-25 13:31
# @Author    :钟新宇

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

from lab4.gradient_11911521 import gradient
from lab4.highboost_11911521 import highboost
from lab4.hist_equ_11911521 import hist_equ
from lab4.laplacian_11911521 import laplacian
from lab4.reduce_SAP_11911521 import reduce_SAP

file = './Q4_2.tif'

if __name__ == '__main__':

    # %% Original_image
    img = np.array(cv.imread(file, cv.IMREAD_GRAYSCALE), dtype=int)

    plt.figure()
    plt.title('Original_image')
    plt.imshow(img, cmap='gray')
    plt.show()
    # %% laplacian_img
    img = np.array(cv.imread(file, cv.IMREAD_GRAYSCALE), dtype=int)
    [laplacian_img, laplacian_mask] = laplacian(img, "operator1", 1)

    plt.figure()
    plt.subplot(121)
    plt.title('laplacian image')
    plt.imshow(laplacian_img, cmap='gray')
    plt.subplot(122)
    plt.title('laplacian_mask')
    plt.imshow(laplacian_mask, cmap='gray')
    plt.show()
    # %% unsharped_img
    img = np.array(cv.imread(file, cv.IMREAD_GRAYSCALE), dtype=int)
    [unsharped_img, unsharped_mask] = highboost(img, 1)

    plt.figure()
    plt.subplot(121)
    plt.title('unsharped_img')
    plt.imshow(unsharped_img, cmap='gray')
    plt.subplot(122)
    plt.title('unsharped_mask')
    plt.imshow(unsharped_mask, cmap='gray')
    plt.show()
    # %% highboost_img
    img = np.array(cv.imread(file, cv.IMREAD_GRAYSCALE), dtype=int)
    [highboost_img, highboost_mask] = highboost(img, 5)

    plt.figure()
    plt.subplot(121)
    plt.title('highboost_img')
    plt.imshow(highboost_img, cmap='gray')
    plt.subplot(122)
    plt.title('highboost_mask')
    plt.imshow(unsharped_mask, cmap='gray')
    plt.show()
    # %% roberts_img
    img = np.array(cv.imread(file, cv.IMREAD_GRAYSCALE), dtype=int)
    [roberts_img, roberts_mask] = gradient(img, "roberts")

    plt.figure()
    plt.subplot(121)
    plt.title('roberts_img')
    plt.imshow(roberts_img, cmap='gray')
    plt.subplot(122)
    plt.title('roberts_mask')
    plt.imshow(roberts_mask, cmap='gray')
    plt.show()
    # %% sobel_img
    img = np.array(cv.imread(file, cv.IMREAD_GRAYSCALE), dtype=int)
    [sobel_img, sobel_mask] = gradient(img, "sobel")

    plt.figure()
    plt.subplot(121)
    plt.title('sobel_img')
    plt.imshow(sobel_img, cmap='gray')
    plt.subplot(122)
    plt.title('sobel_mask')
    plt.imshow(sobel_mask, cmap='gray')
    plt.show()
    # %% combine laplacian & sobel
    img = np.array(cv.imread(file, cv.IMREAD_GRAYSCALE), dtype=int)
    img_a = img
    [img_c, img_b] = laplacian(img_a, "operator1", 1)
    [img_d, _] = gradient(img, "sobel")
    img_e = reduce_SAP(img_d, 5)
    img_f = np.multiply(img_c, img_e)

    img_g = np.array(np.array(img_a) + np.array(img_f), dtype=int)
    img_g = np.array(255 * np.divide(img_g, max(img_g.flat)), dtype=int)
    for i in range(img_g.shape[0]):
        for j in range(img_g.shape[1]):
            if img_g[i, j] < 0: img_g[i, j] = 0

    [img_h, _, _] = hist_equ(img_g)
    # img_h = img_g

    plt.figure()
    plt.subplot(121)
    plt.title('(a) original image')
    plt.imshow(img_a, cmap='gray')
    plt.subplot(122)
    plt.title('(b) Laplacian with scaling')
    plt.imshow(img_b, cmap='gray')
    plt.show()

    plt.figure()
    plt.subplot(121)
    plt.title('(c) Sharped by adding\n(a) and (b)')
    plt.imshow(img_c, cmap='gray')
    plt.subplot(122)
    plt.title('(d) Sobel gradient of (a)')
    plt.imshow(img_d, cmap='gray')
    plt.show()

    plt.figure()
    plt.subplot(121)
    plt.title('(e) Sobel image smoothed\nwith a averaging filter')
    plt.imshow(img_e, cmap='gray')
    plt.subplot(122)
    plt.title('(f) Mask image formed by the\nproduct of (c) and (e)')
    plt.imshow(img_f, cmap='gray')
    plt.show()

    plt.figure()
    plt.subplot(121)
    plt.title('(g) Sharpened image obtained\nby the sum of (a) and (f)')
    plt.imshow(img_g, cmap='gray')
    plt.subplot(122)
    plt.title('(h) Final result obtained by \napplying histogram equali-\nzation to (g)')
    plt.imshow(img_h, cmap='gray')
    plt.show()

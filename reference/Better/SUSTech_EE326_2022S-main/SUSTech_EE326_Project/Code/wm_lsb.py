#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :wm_lsb.py
# @Time      :2022-05-23 12:49
# @Author    :钟新宇

import numpy as np
from Base import readimg, showimg, saveimg, rgb2img, rgb2bgr, dec2bit, bit2dec
from baker import baker_iterate


# def enc(mark, k):
#     return baker_iterate(mark, k=k)


def enc(mark_r, mark_g, mark_b, k):
    mark_r = baker_iterate(mark_r, k=k)
    mark_g = baker_iterate(mark_g, k=k)
    mark_b = baker_iterate(mark_b, k=k)
    return mark_r, mark_g, mark_b


def encode(path, key):
    img_src_path, img_mark_path, mark_src_path, mark_enc_path = path

    # reading data
    img_src, img_r, img_g, img_b = readimg(path=img_src_path)
    mark_src, mark_r, mark_g, mark_b = readimg(path=mark_src_path)

    # get size
    img_src_m, img_src_n, _ = img_src.shape
    mark_m, mark_n, _ = mark_src.shape
    if mark_m * mark_n * 8 > img_src_m * img_src_n:
        print("The source image is too small")

    # dec2bit
    bin_mark_r = np.zeros((mark_m, mark_n, 8), dtype=int)
    bin_mark_g = np.zeros((mark_m, mark_n, 8), dtype=int)
    bin_mark_b = np.zeros((mark_m, mark_n, 8), dtype=int)
    for i in range(mark_m):
        for j in range(mark_n):
            bin_mark_r[i, j, :] = dec2bit(mark_r[i, j])
            bin_mark_g[i, j, :] = dec2bit(mark_g[i, j])
            bin_mark_b[i, j, :] = dec2bit(mark_b[i, j])
    bin_mark_num = mark_m * mark_n * 8
    bin_mark_r2 = np.reshape(bin_mark_r, bin_mark_num)
    bin_mark_g2 = np.reshape(bin_mark_g, bin_mark_num)
    bin_mark_b2 = np.reshape(bin_mark_b, bin_mark_num)

    # padding
    pad_num = img_src_m * img_src_n - mark_m * mark_n * 8
    bin_mark_r2 = np.pad(bin_mark_r2, (0, pad_num), mode='wrap')
    bin_mark_g2 = np.pad(bin_mark_g2, (0, pad_num), mode='wrap')
    bin_mark_b2 = np.pad(bin_mark_b2, (0, pad_num), mode='wrap')

    bin_mark_r2 = np.reshape(bin_mark_r2, (img_src_m, img_src_n))
    bin_mark_g2 = np.reshape(bin_mark_g2, (img_src_m, img_src_n))
    bin_mark_b2 = np.reshape(bin_mark_b2, (img_src_m, img_src_n))

    # mark enc
    bin_mark_r2, bin_mark_g2, bin_mark_b2 = enc(bin_mark_r2, bin_mark_g2, bin_mark_b2, k=key)

    bin_mark_out_rgb = rgb2img(size=(img_src_m, img_src_n), r=bin_mark_r2, g=bin_mark_g2, b=bin_mark_b2)
    bin_mark_out_rgb[bin_mark_out_rgb[:, :, 0] == 1] = 255
    bin_mark_out_rgb[bin_mark_out_rgb[:, :, 1] == 1] = 255
    bin_mark_out_rgb[bin_mark_out_rgb[:, :, 2] == 1] = 255
    showimg(bin_mark_out_rgb, title="bin_mark_rgb")
    saveimg(bin_mark_out_rgb, path=mark_enc_path)

    # lsb
    bin_src_r = np.zeros((img_src_m, img_src_n, 8), dtype=int)
    bin_src_g = np.zeros((img_src_m, img_src_n, 8), dtype=int)
    bin_src_b = np.zeros((img_src_m, img_src_n, 8), dtype=int)
    dec_src_r = np.zeros((img_src_m, img_src_n), dtype=int)
    dec_src_g = np.zeros((img_src_m, img_src_n), dtype=int)
    dec_src_b = np.zeros((img_src_m, img_src_n), dtype=int)

    for i in range(img_src_m):
        for j in range(img_src_n):
            bin_src_r[i, j] = dec2bit(img_r[i, j])
            bin_src_g[i, j] = dec2bit(img_g[i, j])
            bin_src_b[i, j] = dec2bit(img_b[i, j])
            bin_src_r[i, j, 7] = bin_mark_r2[i, j]
            bin_src_g[i, j, 7] = bin_mark_g2[i, j]
            bin_src_b[i, j, 7] = bin_mark_b2[i, j]
            dec_src_r[i, j] = bit2dec(bin_src_r[i, j, :])
            dec_src_g[i, j] = bit2dec(bin_src_g[i, j, :])
            dec_src_b[i, j] = bit2dec(bin_src_b[i, j, :])

    # output
    img_out_rgb = rgb2img(size=(img_src_m, img_src_n), r=dec_src_r, g=dec_src_g, b=dec_src_b)
    showimg(img_out_rgb, title="image with mark")
    saveimg(img_out_rgb, path=img_mark_path)
    return np.asarray((mark_m, mark_n), dtype=int)


def decode(path, size_mark: np.ndarray, key: int):
    img_mark_path, mark_rec_path = path

    # read file
    img_rec, img_rec_r, rec_g, rec_b = readimg(path=img_mark_path)
    # get size
    rec_m, rec_n, _ = img_rec.shape
    mark_m, mark_n = size_mark

    # dec2bit
    bin_rec_r = np.zeros((rec_m, rec_n, 8), dtype=int)
    bin_rec_g = np.zeros((rec_m, rec_n, 8), dtype=int)
    bin_rec_b = np.zeros((rec_m, rec_n, 8), dtype=int)
    bin_mark_rec_r = np.zeros((rec_m, rec_n), dtype=int)
    bin_mark_rec_g = np.zeros((rec_m, rec_n), dtype=int)
    bin_mark_rec_b = np.zeros((rec_m, rec_n), dtype=int)

    for i in range(rec_m):
        for j in range(rec_n):
            bin_rec_r[i, j, :] = dec2bit(img_rec_r[i, j])
            bin_rec_g[i, j, :] = dec2bit(rec_g[i, j])
            bin_rec_b[i, j, :] = dec2bit(rec_b[i, j])
            bin_mark_rec_r[i, j] = bin_rec_r[i, j, 7]
            bin_mark_rec_g[i, j] = bin_rec_g[i, j, 7]
            bin_mark_rec_b[i, j] = bin_rec_b[i, j, 7]

    bin_mark_rec_r, bin_mark_rec_g, bin_mark_rec_b = enc(bin_mark_rec_r, bin_mark_rec_g, bin_mark_rec_b, k=key)

    img_rec_size = rec_m * rec_n
    bin_mark_size = mark_m * mark_n * 8
    mark_pad_num = img_rec_size / bin_mark_size

    bin_mark_rec_r = np.reshape(bin_mark_rec_r, img_rec_size)
    bin_mark_rec_g = np.reshape(bin_mark_rec_g, img_rec_size)
    bin_mark_rec_b = np.reshape(bin_mark_rec_b, img_rec_size)

    # bin_mark_rec_r1 = np.zeros((mark_pad_num, mark_m, mark_n, 8), dtype=int)
    # bin_mark_rec_g1 = np.zeros((mark_pad_num, mark_m, mark_n, 8), dtype=int)
    # bin_mark_rec_b1 = np.zeros((mark_pad_num, mark_m, mark_n, 8), dtype=int)
    # for i in mark_pad_num:
    #     bin_mark_rec_r1[i, :, :] = np.reshape(bin_mark_rec_r[i * bin_mark_size:(i + 1) * bin_mark_size],
    #                                           (mark_m, mark_n, 8))
    #     bin_mark_rec_g1[i, :, :] = np.reshape(bin_mark_rec_g[i * bin_mark_size:(i + 1) * bin_mark_size],
    #                                           (mark_m, mark_n, 8))
    #     bin_mark_rec_b1[i, :, :] = np.reshape(bin_mark_rec_b[i * bin_mark_size:(i + 1) * bin_mark_size],
    #                                           (mark_m, mark_n, 8))

    bin_mark_rec_r1 = bin_mark_rec_r[0:bin_mark_size]
    bin_mark_rec_g1 = bin_mark_rec_g[0:bin_mark_size]
    bin_mark_rec_b1 = bin_mark_rec_b[0:bin_mark_size]

    bin_mark_rec_r1 = np.reshape(bin_mark_rec_r1, (mark_m, mark_n, 8))
    bin_mark_rec_g1 = np.reshape(bin_mark_rec_g1, (mark_m, mark_n, 8))
    bin_mark_rec_b1 = np.reshape(bin_mark_rec_b1, (mark_m, mark_n, 8))

    mark_rec_m, mark_rec_n = mark_m, mark_n
    dec_mark_rec_r = np.zeros((mark_rec_m, mark_rec_n), dtype=int)
    dec_mark_rec_g = np.zeros((mark_rec_m, mark_rec_n), dtype=int)
    dec_mark_rec_b = np.zeros((mark_rec_m, mark_rec_n), dtype=int)
    for i in range(mark_rec_m):
        for j in range(mark_rec_n):
            dec_mark_rec_r[i, j] = bit2dec(bin_mark_rec_r1[i, j, :])
            dec_mark_rec_g[i, j] = bit2dec(bin_mark_rec_g1[i, j, :])
            dec_mark_rec_b[i, j] = bit2dec(bin_mark_rec_b1[i, j, :])

    mark_rec_rgb = rgb2img(size=(mark_rec_m, mark_rec_n), r=dec_mark_rec_r, g=dec_mark_rec_g, b=dec_mark_rec_b)
    showimg(mark_rec_rgb)
    saveimg(mark_rec_rgb, path=mark_rec_path)


if __name__ == "__main__":
    # key: 1024: 2^10, baker 21 times
    k1 = 10
    k2 = 11
    """
    src path
        "./img/img_src.bmp",  # img_src_path
        "./img/img_mark.bmp"  # img_mark_path
        "./img/mark_src.bmp",  # mark_src_path
        "./img/mark_enc.bmp",  # mark_enc_path
        "./img/mark_rec.bmp"  # mark_rec_path
    """

    # # mark size
    # mark_size = np.asarray([256, 256], dtype=int)

    # mark_size = encode(path=["./img/img_src.bmp", "./img/img_mark.bmp", "./img/mark_src.bmp", "./img/mark_enc.bmp"],
    #                    key=k1)
    # decode(path=["./img/img_mark.bmp", "./img/mark_rec.bmp"], size_mark=mark_size, key=k2)

    img_src, _, _, _ = readimg("./img/img_src.bmp")
    img_enc, _, _, _ = readimg("./img/img_mark.bmp")
    img_diff = img_src - img_enc
    # img_diff[img_diff[:, :, :] == 1] = 255
    showimg(img_diff)
    saveimg(img_diff, path="./img/img_diff.bmp")

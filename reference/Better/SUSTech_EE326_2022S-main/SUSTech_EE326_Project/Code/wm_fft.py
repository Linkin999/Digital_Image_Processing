#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :wm_fft.py
# @Time      :2022-05-28 9:30
# @Author    :钟新宇
import cv2
import numpy as np
from numpy.fft import fft2, fftshift, ifft2, ifftshift
from Base import readimg, showimg, saveimg, rgb2img, rgb2bgr, dec2bit, bit2dec
from lib.Filter_Base import normalize
from baker import baker_iterate


def add_mark(img, mark, size):
    img = np.asarray(img, dtype=complex)
    img2 = img.copy()
    m, n = np.asarray(size, dtype=int)

    mark = mark * 20
    mark_flip_1 = np.flip(mark, 0)
    mark_flip_2 = np.flip(mark, 1)
    mark_flip_3 = np.flip(mark_flip_1, 1)
    mark_flip_4 = np.flip(mark_flip_1, 0)
    img2[0:n, 0:m] = np.add(img2[0:n, 0:m], mark, dtype=complex)
    img2[-n:, -m:] = np.add(img2[-n:, -m:], mark_flip_3, dtype=complex)
    img2[-n:, 0:m] = np.add(img2[-n:, 0:m], mark_flip_1, dtype=complex)
    img2[0:n:, -m:] = np.add(img2[0:n:, -m:], mark_flip_2, dtype=complex)
    return img2


def encoder(path):
    img_src_path, img_mark_path, mark_src_path, mark_enc_path = path

    # reading data
    img_src, img_r, img_g, img_b = readimg(path=img_src_path)
    mark_src, mark_r, mark_g, mark_b = readimg(path=mark_src_path)

    # get size
    img_src_m, img_src_n, _ = img_src.shape
    mark_m, mark_n, _ = mark_src.shape

    # fft2
    img_r_fft = fftshift(fft2(img_r))
    img_g_fft = fftshift(fft2(img_g))
    img_b_fft = fftshift(fft2(img_b))

    # log magnitude
    img_r_mag = normalize(np.log(np.abs(img_r_fft)))
    img_g_mag = normalize(np.log(np.abs(img_g_fft)))
    img_b_mag = normalize(np.log(np.abs(img_b_fft)))
    img_src_mag = normalize(rgb2img(size=(img_src_m, img_src_n), r=img_r_mag, g=img_g_mag, b=img_b_mag), dtype=int)
    showimg(img=img_src_mag, title="img_src_mag")
    saveimg(img_src_mag, path="./img/img_src_fft_mag.bmp")

    # angle
    img_r_ang = np.angle(img_r_fft)
    img_g_ang = np.angle(img_g_fft)
    img_b_ang = np.angle(img_b_fft)

    # add watermark
    enc_r_fft = add_mark(img_r_fft, mark=mark_r, size=(mark_m, mark_n))
    enc_g_fft = add_mark(img_g_fft, mark=mark_g, size=(mark_m, mark_n))
    enc_b_fft = add_mark(img_b_fft, mark=mark_b, size=(mark_m, mark_n))
    # magnitude
    enc_r_mag = normalize(np.log(np.abs(enc_r_fft)))
    enc_g_mag = normalize(np.log(np.abs(enc_g_fft)))
    enc_b_mag = normalize(np.log(np.abs(enc_b_fft)))
    enc_mag = normalize(rgb2img(size=(img_src_m, img_src_n), r=enc_r_mag, g=enc_g_mag, b=enc_b_mag), dtype=int)
    showimg(img=enc_mag, title="enc_mag")
    saveimg(enc_mag, path="./img/enc_fft_mag.bmp")
    # ifft2
    enc_r_ifft = (ifft2(enc_r_fft))
    enc_g_ifft = (ifft2(enc_g_fft))
    enc_b_ifft = (ifft2(enc_b_fft))

    img_enc_r = np.asarray(np.abs(enc_r_ifft), dtype=int)
    img_enc_g = np.asarray(np.abs(enc_g_ifft), dtype=int)
    img_enc_b = np.asarray(np.abs(enc_b_ifft), dtype=int)
    img_enc = normalize(rgb2img(size=(img_src_m, img_src_n), r=img_enc_r, g=img_enc_g, b=img_enc_b), dtype=int)
    showimg(img=img_enc, title="img_enc")
    saveimg(img_enc, path=img_mark_path)


def decoder(path):
    img_mark_path = path

    # reading data
    img_enc, enc_r, enc_g, enc_b = readimg(path=img_mark_path)

    # get size
    img_m, img_n, _ = img_enc.shape

    # fft2
    enc_r_fft = fftshift(fft2(enc_r))
    enc_g_fft = fftshift(fft2(enc_g))
    enc_b_fft = fftshift(fft2(enc_b))

    # log magnitude
    enc_r_mag = normalize(np.log(np.abs(enc_r_fft)))
    enc_g_mag = normalize(np.log(np.abs(enc_g_fft)))
    enc_b_mag = normalize(np.log(np.abs(enc_b_fft)))
    enc_mag = normalize(rgb2img(size=(img_m, img_n), r=enc_r_mag, g=enc_g_mag, b=enc_b_mag), dtype=int)
    showimg(img=enc_mag, title="enc_mag")
    saveimg(enc_mag, path="./img/img_rec_mag.bmp")


if __name__ == '__main__':
    try:
        """
        src path
            "./img/img_src.bmp",  # img_src_path
            "./img/img_mark.bmp"  # img_mark_path
            "./img/mark_src.bmp",  # mark_src_path
            "./img/mark_enc.bmp",  # mark_enc_path
            "./img/mark_rec.bmp"  # mark_rec_path
        """
        encode_path = np.asarray(["./img/img_src.bmp", "./img/img_mark.bmp", "./img/mark_src.bmp", "./img/mark_enc.bmp"], dtype=str)
        decode_path = "./img/img_mark.bmp"
        encoder(encode_path)
        decoder(decode_path)
        print("end")
    except KeyboardInterrupt:
        pass

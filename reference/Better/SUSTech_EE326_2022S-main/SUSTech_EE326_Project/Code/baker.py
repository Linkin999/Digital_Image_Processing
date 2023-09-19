#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :baker.py
# @Time      :2022-05-23 18:31
# @Author    :钟新宇
import numpy as np


def baker_stretch(array):
    """
    Program a baker_stretch(array) function that returns the new array obtained by “stretching”
    the input table.

    Stretching. The principle is as follows: the first two lines (each with a length of n) produce a single
    line with a length of 2n. We mix the values of each line by alternating an upper element and a
    lower element.

    Formulas. An element at position (i, j) of the target array, corresponds to an element (2i, j//2) (if j
    is even) or (2i + 1, j//2) (if j is odd) of the source array, with here 0 ⩽ i < n
    2 and 0 ⩽ j < 2n.

    :param array: The input array
    :return: The output array after baker_stretch
    """
    array = np.asarray(array, dtype=int)
    m, n = array.shape
    array_out = np.zeros((int(m / 2), int(n * 2)), dtype=int)
    for i in range(int(m / 2)):
        for j in range(int(n * 2)):
            if j % 2 == 0:
                array_out[i, j] = array[2 * i, j // 2]
            else:
                array_out[i, j] = array[2 * i + 1, j // 2]
    # print("strtch")
    # print(array_out)
    return array_out


def baker_fold(array):
    """
    Program a baker_fold(array) function that returns the table obtained by “folding” the input
    table.

    Fold. The principle is as follows: the right part of a stretched array is turned upside down, then
    added under the left part. Starting from a n/2 × 2n array you get an n × n array.

    Formulas. For 0 ⩽ i < n/2 and 0 ⩽ j < n the elements in position (i, j) of the array are kept in
    place. For n/2 ⩽ i < n and 0 ⩽ j < n an element of the array (i, j) corresponds to an element
    (n/2 − i − 1,2n − 1 − j) of the source array.

    :param array: The input array after baker_stretch
    :return: The output array after baker_fold
    """
    array = np.asarray(array, dtype=int)
    m, n = array.shape
    array_out = np.zeros((int(m * 2), int(n / 2)), dtype=int)
    for i in range(0, m):
        for j in range(int(n / 2)):
            array_out[i, j] = array[i, j]
    for i in range(m, m * 2):
        for j in range(int(n / 2)):
            array_out[i, j] = array[m - i - 1, n - 1 - j]
    # print("fold")
    # print(array_out)
    return array_out


def baker_iterate(array, k):
    """
    Program a baker_iterate(array,k) function that returns the table calculated after k iterations
    of baker’s transformation.

    Caution! It sometimes takes many iterations to get back to the original image. For example when
    n = 4, we return to the starting image after k = 5 iterations; when n = 256 it takes k = 17.
    Conjecture a return value in the case where n is a power of 2. However, for n = 10, you need
    k = 56920 iterations!

    :param array:
    :param k:
    :return:
    """
    for i in range(k):
        print("iterate No.%d" % (i + 1))
        array = baker(array)
        print(array)

    return array


def baker(array):
    array_stretch = baker_stretch(array)
    array_fold = baker_fold(array_stretch)
    return array_fold


if __name__ == '__main__':
    try:
        array_in = np.asarray([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], dtype=int)
        print(array_in)
        # array_baker = baker(array_in)
        array_baker = baker_iterate(array_in, k=5)
    except KeyboardInterrupt:
        pass

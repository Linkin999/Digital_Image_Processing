import numpy as np
import cv2 as cv


def nearest_11911521(input_file, dim):
    src_img = cv.imread(input_file, cv.IMREAD_GRAYSCALE)
    src_h, src_w = src_img.shape
    tag_h, tag_w = int(dim[0]), int(dim[1])
    tag_img = np.zeros((tag_h, tag_w))
    factor_y, factor_x = (src_h - 1) / (tag_h - 1), (src_w - 1) / (tag_w - 1)

    for i in range(tag_h):
        for j in range(tag_w):
            src_y = int(round(i * factor_y))
            src_x = int(round(j * factor_x))
            tag_img[i][j] = src_img[src_y][src_x]
    return tag_img


if __name__ == '__main__':
    path = 'rice.tif'
    dim = [256 * 1.1, 256 * 1.1]
    newfile1 = nearest_11911521(path, dim)
    cv.imwrite('enlarged_nearest_11911521.tif', newfile1)
    cv.imwrite('enlarged_nearest_11911521.png', newfile1)

    path = 'rice.tif'
    dim = [256 * 0.9, 256 * 0.9]
    newfile2 = nearest_11911521(path, dim)
    cv.imwrite('shrunk_nearest_11911521.tif', newfile2)
    cv.imwrite('shrunk_nearest_11911521.png', newfile2)

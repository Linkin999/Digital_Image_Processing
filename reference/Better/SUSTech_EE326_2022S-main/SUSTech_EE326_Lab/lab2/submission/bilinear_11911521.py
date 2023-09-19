import numpy as np
import cv2 as cv


def bilinear_11911521(input_file, dim):
    src_img = cv.imread(input_file, cv.IMREAD_GRAYSCALE)
    src_h, src_w = src_img.shape
    tag_h, tag_w = int(dim[0]), int(dim[1])
    tag_img = np.zeros((tag_h, tag_w))
    factor_y, factor_x = (src_h - 1) / (tag_h - 1), (src_w - 1) / (tag_w - 1)

    for i in range(tag_h):
        for j in range(tag_w):
            src_y = i * factor_y
            src_x = j * factor_x

            y1 = int(np.floor(src_y))
            y2 = int(np.ceil(src_y))
            x1 = int(np.floor(src_x))
            x2 = int(np.ceil(src_x))

            diff_y = src_y - y1
            diff_x = src_x - x1

            weight = [(1 - diff_y) * (1 - diff_x), diff_y * (1 - diff_x), (1 - diff_y) * diff_x, diff_y * diff_x]
            q = [src_img[y1][x1], src_img[y2][x1], src_img[y1][x2], src_img[y2][x2]]

            for k in range(0, 3):
                tag_img[i][j] += round(weight[k] * q[k])

    return tag_img


if __name__ == '__main__':
    path = 'rice.tif'
    dim = [256 * 1.1, 256 * 1.1]
    newfile1 = bilinear_11911521(path, dim)
    cv.imwrite('enlarged_bilinear_11911521.tif', newfile1)
    cv.imwrite('enlarged_bilinear_11911521.png', newfile1)

    path = 'rice.tif'
    dim = [256 * 0.9, 256 * 0.9]
    newfile2 = bilinear_11911521(path, dim)
    cv.imwrite('shrunk_bilinear_11911521.tif', newfile2)
    cv.imwrite('shrunk_bilinear_11911521.png', newfile2)

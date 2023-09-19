import numpy as np
import cv2 as cv
from scipy.interpolate import interp2d


def bicubic_11911521(input_file, dim):
    src_img = cv.imread(input_file, cv.IMREAD_GRAYSCALE)
    src_h, src_w = src_img.shape
    tag_h, tag_w = int(dim[0]), int(dim[1])

    interpolator = interp2d(range(src_h), range(src_w), src_img, kind='cubic')
    tag_y = np.linspace(0, src_h - 1, num=tag_h)
    tag_x = np.linspace(0, src_w - 1, num=tag_w)
    tag_img = interpolator(tag_y, tag_x)

    return tag_img


if __name__ == '__main__':
    path = 'rice.tif'
    dim = [256 * 1.1, 256 * 1.1]
    newfile1 = bicubic_11911521(path, dim)
    cv.imwrite('enlarged_bicubic_11911521.tif', newfile1)
    cv.imwrite('enlarged_bicubic_11911521.png', newfile1)

    path = 'rice.tif'
    dim = [256 * 0.9, 256 * 0.9]
    newfile2 = bicubic_11911521(path, dim)
    cv.imwrite('shrunk_bicubic_11911521.tif', newfile2)
    cv.imwrite('shrunk_bicubic_11911521.png', newfile2)

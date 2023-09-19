import cv2
import numpy as np
from matplotlib import pyplot as plt


def reduce_SAP(image, n_size):
    img = np.array(image)
    m, n = img.shape
    local_len = int((n_size - 1) / 2)

    output_image = np.array(np.zeros((m, n)))

    if m - 1 - local_len <= local_len or n - 1 - local_len <= local_len:
        print("The parameter k is to large.")

    for i in range(local_len, m - local_len):
        for j in range(local_len, n - local_len):
            output_image[i][j] = np.median(img[i - local_len:i + local_len + 1, j - local_len:j + local_len + 1])

    output_image = np.array(255 * np.divide(output_image, max(output_image.flat)), dtype=int)
    for i in range(m):
        for j in range(n):
            if output_image[i, j] < 0: output_image[i, j] = 0
    return output_image


if __name__ == '__main__':
    try:
        path = 'Q3_4.tif'
        image = cv2.imread("Q3_4.tif.", cv2.IMREAD_GRAYSCALE)
        cv2.imwrite("Q3_4.png", image)
        out_img = reduce_SAP(path, 5)
        cv2.imwrite("Q3_4_11911521.tif", out_img)
        cv2.imwrite("Q3_4_11911521.png", out_img)
        image = cv2.imread("Q3_4_11911521.tif", cv2.IMREAD_GRAYSCALE)
        cv2.imshow('Q3_4_11911521', image)
        cv2.waitKey(0)

    except KeyboardInterrupt:
        pass

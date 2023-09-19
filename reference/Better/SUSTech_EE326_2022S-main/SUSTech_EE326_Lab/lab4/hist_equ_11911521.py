import cv2
import matplotlib.pyplot as plt
import numpy as np


def hist_equ(img):
    m, n = img.shape
    L = 256
    bins = range(L + 1)
    input_hist, _ = np.histogram(img.flat, bins=bins, density=True)

    s = np.array(np.zeros(256), dtype=int)
    for i in range(L):
        s[i] = (L - 1) * sum(input_hist[:i + 1])

    output_image = np.array(np.zeros((m, n)), dtype=int)
    for i in range(m):
        for j in range(n):
            output_image[i, j] = round(s[round(img[i, j])])

    output_image = np.array(255 * np.divide(output_image, max(output_image.flat)), dtype=int)
    for i in range(m):
        for j in range(n):
            if output_image[i, j] < 0: output_image[i, j] = 0
    output_hist, _ = np.histogram(output_image.flat, bins=bins, density=True)

    return output_image, output_hist, input_hist


if __name__ == '__main__':
    image = cv2.imread("Q3_1_1.tif.", cv2.IMREAD_GRAYSCALE)
    cv2.imwrite("Q3_1_1.png", image)

    image = cv2.imread("Q3_1_2.tif", cv2.IMREAD_GRAYSCALE)
    cv2.imwrite("Q3_1_2.png", image)
    # Q3_1_1
    path = 'Q3_1_1.tif'
    [out_img_1, out_hist_1, in_hist_1] = hist_equ(path)
    cv2.imwrite("Q3_1_1_11911521.tif", out_img_1)
    cv2.imwrite("Q3_1_1_11911521.png", out_img_1)

    image = cv2.imread("Q3_1_1_11911521.tif", cv2.IMREAD_GRAYSCALE)
    cv2.imshow('image', image)
    cv2.waitKey(0)
    # Q3_1_2
    path = 'Q3_1_2.tif'
    [out_img_2, out_hist_2, in_hist_2] = hist_equ(path)
    cv2.imwrite("Q3_1_2_11911521.tif", out_img_2)
    cv2.imwrite("Q3_1_2_11911521.png", out_img_2)
    image = cv2.imread("Q3_1_2_11911521.tif", cv2.IMREAD_GRAYSCALE)
    cv2.imshow('image', image)
    cv2.waitKey(0)

    plt.bar(range(256), out_hist_1)
    plt.title("out_hist_1")
    plt.savefig("Q3_1_out_hist_1.png")
    plt.show()
    plt.bar(range(256), in_hist_1)
    plt.title("in_hist_1")
    plt.savefig("Q3_1_in_hist_1.png")
    plt.show()

    plt.bar(range(256), out_hist_2)
    plt.title("out_hist_2")
    plt.savefig("Q3_1_out_hist_2.png")
    plt.show()
    plt.bar(range(256), in_hist_2)
    plt.title("in_hist_2")
    plt.savefig("Q3_1_in_hist_2.png")
    plt.show()

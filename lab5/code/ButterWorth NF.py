import cv2
import numpy as np


def Get_Spectrum(input_image):
    M ,N= input_image.shape
    P = 2 * M
    Q = 2 * N
    zp_image = np.zeros([P, Q, 3])
    for x in range(0, M):
        for y in range(0, N):
            zp_image[x, y, :] = input_image[x, y, :] * ((-1) ** (x + y))

    Filter_matrix = np.zeros([P, Q, 3], dtype=complex)
    Filter_matrix[:, :, 0] = np.fft.fft2(zp_image[:, :, 0])
    Filter_matrix[:, :, 1] = np.fft.fft2(zp_image[:, :, 1])
    Filter_matrix[:, :, 2] = np.fft.fft2(zp_image[:, :, 2])

    Filter_abs = np.zeros([P, Q, 3])
    Filter_abs[:, :, 0] = 20 * np.log10(abs(Filter_matrix[:, :, 0]))
    Filter_abs[:, :, 1] = 20 * np.log10(abs(Filter_matrix[:, :, 1]))
    Filter_abs[:, :, 2] = 20 * np.log10(abs(Filter_matrix[:, :, 2]))
    return Filter_abs


def ButterWorth_Nortch(input_image, D0, n):
    M, N = input_image.shape
    P = 2 * M
    Q = 2 * N
    zero_pro_image = np.zeros([P, Q, 3])
    for x in range(0, M):
        for y in range(0, N):
            zero_pro_image[x, y, :] = input_image[x, y, :] * ((-1) ** (x + y))

    Filter_matrix = np.zeros([P, Q, 3], dtype=complex)
    Filter_matrix[:, :, 0] = np.fft.fft2(zero_pro_image[:, :, 0])
    Filter_matrix[:, :, 1] = np.fft.fft2(zero_pro_image[:, :, 1])
    Filter_matrix[:, :, 2] = np.fft.fft2(zero_pro_image[:, :, 2])

    Filter_abs = np.zeros([P, Q, 3])
    Filter_abs[:, :, 0] = 20 * np.log10(abs(Filter_matrix[:, :, 0]))
    Filter_abs[:, :, 1] = 20 * np.log10(abs(Filter_matrix[:, :, 1]))
    Filter_abs[:, :, 2] = 20 * np.log10(abs(Filter_matrix[:, :, 2]))

    # BNF_operator
    H_Matrix = np.zeros([P, Q])
    Output_Image_H = np.zeros([P, Q, 3])
    D1 = [79, 222]
    D2 = [87, 109]
    D3 = [162, 222]
    D4 = [169, 109]  # these four arrays are found by F_F.tif
    D1_1 = [P - 79, Q - 222]
    D2_1 = [P - 87, Q - 109]
    D3_1 = [P - 162, Q - 222]
    D4_1 = [P - 169, Q - 109]

    for u in range(0, P):
        for v in range(0, Q):
            if [u, v] not in [D1, D2, D3, D4, D1_1, D2_1, D3_1, D4_1]:
                H_matrix_1_k = 1 / (1 + (D0 ** 2 / ((u - D1[0]) ** 2 + (v - D1[1]) ** 2)) ** n)
                H_matrix_1_kT = 1 / (1 + (D0 ** 2 / ((u - D1_1[0]) ** 2 + (v - D1_1[1]) ** 2)) ** n)
                H_matrix_2_k = 1 / (1 + (D0 ** 2 / ((u - D2[0]) ** 2 + (v - D2[1]) ** 2)) ** n)
                H_matrix_2_kT = 1 / (1 + (D0 ** 2 / ((u - D2_1[0]) ** 2 + (v - D2_1[1]) ** 2)) ** n)
                H_matrix_3_k = 1 / (1 + (D0 ** 2 / ((u - D3[0]) ** 2 + (v - D3[1]) ** 2)) ** n)
                H_matrix_3_kT = 1 / (1 + (D0 ** 2 / ((u - D3_1[0]) ** 2 + (v - D3_1[1]) ** 2)) ** n)
                H_matrix_4_k = 1 / (1 + (D0 ** 2 / ((u - D4[0]) ** 2 + (v - D4[1]) ** 2)) ** n)
                H_matrix_4_kT = 1 / (1 + (D0 ** 2 / ((u - D4_1[0]) ** 2 + (v - D4_1[1]) ** 2)) ** n)
                H_Matrix[
                    u, v] = H_matrix_1_k * H_matrix_1_kT * H_matrix_2_k * H_matrix_2_kT * H_matrix_3_k * H_matrix_3_kT * H_matrix_4_k * H_matrix_4_kT
            Output_Image_H[u, v, 0] = int(255 * H_Matrix[u, v])
            Output_Image_H[u, v, 1] = int(255 * H_Matrix[u, v])
            Output_Image_H[u, v, 2] = int(255 * H_Matrix[u, v])

    G_matrix = np.zeros([P, Q, 3], dtype=complex)
    G_matrix[:, :, 0] = np.multiply(H_Matrix, Filter_matrix[:, :, 0])
    G_matrix[:, :, 1] = np.multiply(H_Matrix, Filter_matrix[:, :, 1])
    G_matrix[:, :, 2] = np.multiply(H_Matrix, Filter_matrix[:, :, 2])

    g_p = np.zeros([P, Q, 3], dtype=complex)
    g_p[:, :, 0] = np.fft.ifft2(G_matrix[:, :, 0])
    g_p[:, :, 1] = np.fft.ifft2(G_matrix[:, :, 1])
    g_p[:, :, 2] = np.fft.ifft2(G_matrix[:, :, 2])

    for x in range(0, P):
        for y in range(0, Q):
            g_p[x, y, :] = np.real(g_p[x, y, :]) * ((-1) ** (x + y))

    g_final = np.zeros([M, N, 3])
    for x in range(0, M - 1):
        for y in range(0, N - 1):
            g_final[x, y, :] = np.real(g_p[x, y, :])
    return [Filter_abs, g_final, Output_Image_H]


input_file = cv2.imread('Q5_3.tif')

[output_F, output_image, output_H_F] = ButterWorth_Nortch(input_file, 30, 4)
cv2.imwrite('F_abs.tif', output_F)
cv2.imwrite('ButterWorth NF.tif', output_image)
cv2.imwrite('output_H.tif', output_H_F)

out_im_F = Get_Spectrum(output_image)
cv2.imwrite('OUT_abs.tif', out_im_F)


# to find notch
av = np.zeros([output_F.shape[0], output_F.shape[1]])
tot = 0
num = 0

for i in range(0, output_F.shape[0]):
    for j in range(0, output_F.shape[1]):
        av[i, j] = (output_F[i, j, 0] + output_F[i, j, 1] + output_F[i, j, 2]) / 3
        tot += av[i, j]
        num += 1
aver = tot / num

F_F = np.zeros([output_F.shape[0], output_F.shape[1], 3])

for i in range(0, output_F.shape[0]):
    for j in range(0, output_F.shape[1]):
        if av[i, j] > 110:
            F_F[i, j, :] = [255, 255, 255]
cv2.imwrite('F_F.tif', F_F)

if __name__=='__main__':
    print('abatewsois1')
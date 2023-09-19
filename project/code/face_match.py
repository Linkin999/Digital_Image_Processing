from matplotlib import pyplot as plt
import numpy as np
import cv2
import time
from skimage import feature as ft

from Hog_feature import Hog_descriptor


##Bilateral filter, grayscale
def Gauss_kernel(size,sigma_d):##size最好为奇数
    radius=int((size-1)/2)
    kernel_gauss=np.zeros([size,size],dtype=np.float64)
    factor_gauss=2*sigma_d*sigma_d
    for i in range(-radius,radius+1):
        for j in range(-radius,radius+1):
            kernel_gauss[i+radius][j+radius]=np.exp(-(i*i+j*j)/factor_gauss)
    return kernel_gauss

def Bilateral_filtering(input_image,size,sigma_d,sigma_r):
    H,W=input_image.shape
    output_image=np.zeros([H,W],dtype=np.float64)

    radius=int((size-1)/2)

    ##pad image into size (H+1)×(W+1), fill with edge values
    pad_image=np.pad(input_image,((radius,radius),(radius,radius)),'edge')
    pad_image=np.array(pad_image,dtype=np.float64)

    #calculate value
    for i in range(radius,radius+H):
        for j in range(radius,radius+W):

            ##Gausss kernel
            kernel_gauss=Gauss_kernel(size,sigma_d)

            #space kernel
            space_kernel=np.zeros([size,size],dtype=np.float64)
            factor_space=2*sigma_r*sigma_r
            value_kernel=np.zeros([size,size],dtype=np.float64)
            for k in range(-radius,radius+1):
                for l in range(-radius,radius+1):
                    space_kernel[k+radius][l+radius]=np.exp(-(np.power((float(abs(pad_image[i][j]-pad_image[i+k][j+l]))),2))/factor_space)
                    value_kernel[k+radius][l+radius]=pad_image[i+k][j+l]

            #w ,size×size
            kernel_w=kernel_gauss*space_kernel


            output_image[i-radius][j-radius]=sum(sum(value_kernel*kernel_w))/sum(sum(kernel_w))

    output_image=np.array(output_image,dtype=np.uint8)

    return output_image




if  __name__=='__main__':
    """a=np.array([(3,2,1),(6,5,4),(9,8,7)])
    b=np.array([(1,2,3),(4,5,6),(7,8,9)])
    print(a)
    print(b)
    print(a*b)
    c=np.sum(a*b)
    print(c)
    d=np.power(a,2)
    print(d)
    e=np.power(np.sum(np.power(a,2)),0.5)*np.power(np.sum(np.power(a,2)),0.5)
    print(e)"""
    

    img=cv2.imread('Q4_2.tif',cv2.IMREAD_GRAYSCALE)
    cv2.imshow('Q4_2',img)
    print(img.shape)
    #out=gamma_normalize(img)
    #cv2.imshow('processed',out)
    hog=Hog_descriptor(img,cell_size=8,bin_size=9)
    hog_vector,hog_image=hog.extract()
    plt.imshow(hog_image,cmap=plt.cm.gray)
    plt.show()
    print(hog_vector.shape)
    
    
    out2=ft.hog(img,orientations=9,pixels_per_cell=[8,8],cells_per_block=[2,2],visualize=True)
    plt.imshow(out2[1],cmap=plt.cm.gray)
    plt.show()
    print(out2[0].shape)


    """time_start = time.perf_counter()
    out1=Bilateral_filtering(img,3,11,11)
    time_end = time.perf_counter()
    run_time = time_end - time_start
    print("运行时长：", run_time)
    cv2.imshow('Q4_2_processed_mymethod',out1)

    time_start2 = time.perf_counter()
    out2=cv2.bilateralFilter(img,3,11,11)
    time_end2 = time.perf_counter()
    run_time2 = time_end2 - time_start2
    print("运行时长：", run_time2)
    cv2.imshow('Q4_2_processed_cv2',out2)

    cv2.imshow('Q4_2_processed_difference1',out2-out1)
    print(out2-out1)

    cv2.imshow('Q4_2_processed_difference2',out1-img)
    print(out1-img)

    cv2.imshow('Q4_2_processed_difference3',out2-img)
    print(out2-img)"""

    



    cv2.waitKey(0)
import time
from PIL import Image
import numpy as np
import cv2
import math

def full_inverse_filter(input_image,k):
    input_image_DFT=np.fft.fft2(input_image)
    input_image_center=np.fft.fftshift(input_image_DFT)
    row,col=input_image_center.shape
    H=np.zeros([row,col],dtype=np.complex128)
    for i in range (0,row):
        for j in range(0,col):
            d=np.power(i-row/2,2)+np.power(j-col/2,2)
            H[i,j]=np.exp(-(k*(np.power(d,5/6))))
    output_image=input_image_center/H
    output_image=np.fft.ifftshift(np.fft.ifft2(output_image))
    output_image=np.real(output_image)

    a=np.max(output_image)
    b=np.min(output_image)
    for i in range(0,output_image.shape[0]):
        for j in range(0,output_image.shape[1]):
            output_image[i,j]=int((output_image[i,j]-b)*255/(a-b))
    
    output_image=np.array(output_image,dtype=np.uint8)
    return output_image

def limited_inverse_filter(input_image,radius,k,order):
    input_image_DFT=np.fft.fft2(input_image)
    input_image_center=np.fft.fftshift(input_image_DFT)
    row,col=input_image_center.shape

    H=np.zeros([row,col],dtype=np.complex128)
    for i in range (0,row):
        for j in range(0,col):
            d=np.power(i-row/2,2)+np.power(j-col/2,2)
            H[i,j]=np.exp(-(k*(np.power(d,5/6))))
    
    #Butterworth lowpass filter
    H_Butterworth=np.zeros([row,col],dtype=np.complex128)
    for i in range (0,row):
        for j in range(0,col):
            d=np.power(i-row/2,2)+np.power(j-col/2,2)
            H_Butterworth[i,j]=1/(1+np.power(d/(np.power(radius,2)),order))
    
    output_image=input_image_center*H_Butterworth/H

    output_image=np.fft.ifft2(np.fft.ifftshift(output_image))
    output_image=np.real(output_image)

    a=np.max(output_image)
    b=np.min(output_image)
    for i in range(0,output_image.shape[0]):
        for j in range(0,output_image.shape[1]):
            output_image[i,j]=int((output_image[i,j]-b)*255/(a-b))
    
    output_image=np.array(output_image,dtype=np.uint8)
    return output_image



def Wiener_filter(input_image,K):
    input_image_DFT=np.fft.fft2(input_image)
    input_image_center=np.fft.fftshift(input_image_DFT)

    row,col=input_image_center.shape

    H=np.zeros([row,col],dtype=np.complex128)
    for i in range (0,row):
        for j in range(0,col):
            d=np.power(i-row/2,2)+np.power(j-col/2,2)
            H[i,j]=np.exp(-(0.0025*(np.power(d,5/6))))
    
    buf=np.conj(H)*H
    output_image=input_image_center/H*(buf/(buf+K))

    output_image=np.fft.ifft2(np.fft.ifftshift(output_image))
    output_image=np.real(output_image)

    a=np.max(output_image)
    b=np.min(output_image)
    for i in range(0,output_image.shape[0]):
        for j in range(0,output_image.shape[1]):
            output_image[i,j]=int((output_image[i,j]-b)*255/(a-b))
    
    output_image=np.array(output_image,dtype=np.uint8)
    return output_image


if  __name__=='__main__':
    img=cv2.imread('Q6_2.tif',cv2.IMREAD_GRAYSCALE)
    cv2.imshow('Q6_2',img)

    time_start = time.perf_counter()
    out1=full_inverse_filter(img,0.0025)
    time_end = time.perf_counter()
    run_time = time_end - time_start
    print("运行时长：", run_time)
    cv2.imshow('Q6_2_restore_full_inverse_filter',out1)
    savedimage1=Image.fromarray(out1)
    savedimage1.save('Q6_2_restore_full_inverse_filter.tif')

    out2_40_10=limited_inverse_filter(img,40,0.0025,10)
    cv2.imshow('Q6_2_restore_limited_inverse_filter_40_10',out2_40_10)
    savedimage2_40_10=Image.fromarray(out2_40_10)
    savedimage2_40_10.save('Q6_2_restore_limited_inverse_filter_40_10.tif')

    out2_30_10=limited_inverse_filter(img,30,0.0025,10)
    cv2.imshow('Q6_2_restore_limited_inverse_filter_30_10',out2_30_10)
    savedimage2_30_10=Image.fromarray(out2_30_10)
    savedimage2_30_10.save('Q6_2_restore_limited_inverse_filter_30_10.tif')

    out2_50_10=limited_inverse_filter(img,50,0.0025,10)
    cv2.imshow('Q6_2_restore_limited_inverse_filter_50_10',out2_50_10)
    savedimage2_50_10=Image.fromarray(out2_50_10)
    savedimage2_50_10.save('Q6_2_restore_limited_inverse_filter_50_10.tif')

    out2_60_10=limited_inverse_filter(img,60,0.0025,10)
    cv2.imshow('Q6_2_restore_limited_inverse_filter_60_10',out2_60_10)
    savedimage2_60_10=Image.fromarray(out2_60_10)
    savedimage2_60_10.save('Q6_2_restore_limited_inverse_filter_60_10.tif')

    out2_70_10=limited_inverse_filter(img,70,0.0025,10)
    cv2.imshow('Q6_2_restore_limited_inverse_filter_70_10',out2_70_10)
    savedimage2_70_10=Image.fromarray(out2_70_10)
    savedimage2_70_10.save('Q6_2_restore_limited_inverse_filter_70_10.tif')

    out2_80_10=limited_inverse_filter(img,80,0.0025,10)
    cv2.imshow('Q6_2_restore_limited_inverse_filter_80_10',out2_80_10)
    savedimage2_80_10=Image.fromarray(out2_80_10)
    savedimage2_80_10.save('Q6_2_restore_limited_inverse_filter_80_10.tif')

    out2_60_5=limited_inverse_filter(img,60,0.0025,5)
    cv2.imshow('Q6_2_restore_limited_inverse_filter_60_5',out2_60_5)
    savedimage2_60_5=Image.fromarray(out2_60_5)
    savedimage2_60_5.save('Q6_2_restore_limited_inverse_filter_60_5.tif')

    time_start1 = time.perf_counter()
    out2_60_10=limited_inverse_filter(img,60,0.0025,10)
    time_end1 = time.perf_counter()
    run_time1 = time_end1 - time_start1
    print("运行时长：", run_time1)
    cv2.imshow('Q6_2_restore_limited_inverse_filter_60_10',out2_60_10)
    savedimage2_60_10=Image.fromarray(out2_60_10)
    savedimage2_60_10.save('Q6_2_restore_limited_inverse_filter_60_10.tif')
    
    out2_60_15=limited_inverse_filter(img,60,0.0025,15)
    cv2.imshow('Q6_2_restore_limited_inverse_filter_60_15',out2_60_15)
    savedimage2_60_15=Image.fromarray(out2_60_15)
    savedimage2_60_15.save('Q6_2_restore_limited_inverse_filter_60_15.tif')

    out2_60_20=limited_inverse_filter(img,60,0.0025,20)
    cv2.imshow('Q6_2_restore_limited_inverse_filter_60_20',out2_60_20)
    savedimage2_60_20=Image.fromarray(out2_60_20)
    savedimage2_60_20.save('Q6_2_restore_limited_inverse_filter_60_20.tif')

    out3_1=Wiener_filter(img,0.0025)
    cv2.imshow('Q6_2_restore_Wiener_filter_0.0025',out3_1)
    savedimage3_1=Image.fromarray(out3_1)
    savedimage3_1.save('Q6_2_restore_Wiener_filter_0.0025.tif')

    out3_2=Wiener_filter(img,0.1)
    cv2.imshow('Q6_2_restore_Wiener_filter_0.1',out3_2)
    savedimage3_2=Image.fromarray(out3_2)
    savedimage3_2.save('Q6_2_restore_Wiener_filter_0.1.tif')

    out3_3=Wiener_filter(img,0.01)
    cv2.imshow('Q6_2_restore_Wiener_filter_0.01',out3_3)
    savedimage3_3=Image.fromarray(out3_3)
    savedimage3_3.save('Q6_2_restore_Wiener_filter_0.01.tif')

    time_start2 = time.perf_counter()
    out3_4=Wiener_filter(img,0.001)
    time_end2 = time.perf_counter()
    run_time2 = time_end2 - time_start2
    print("运行时长：", run_time2)
    cv2.imshow('Q6_2_restore_Wiener_filter_0.001',out3_4)
    savedimage3_4=Image.fromarray(out3_4)
    savedimage3_4.save('Q6_2_restore_Wiener_filter_0.001.tif')

    out3_5=Wiener_filter(img,0.0001)
    cv2.imshow('Q6_2_restore_Wiener_filter_0.0001',out3_5)
    savedimage3_5=Image.fromarray(out3_5)
    savedimage3_5.save('Q6_2_restore_Wiener_filter_0.0001.tif')

    out3_6=Wiener_filter(img,0.00001)
    cv2.imshow('Q6_2_restore_Wiener_filter_0.00001',out3_6)
    savedimage3_6=Image.fromarray(out3_6)
    savedimage3_6.save('Q6_2_restore_Wiener_filter_0.00001.tif')
    
    cv2.waitKey(0)

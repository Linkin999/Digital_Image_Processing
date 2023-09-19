import time
from PIL import Image
import numpy as np
import cv2
import math


def Adaptive_Median_Filter(input_image,Smax,Sinit):
    H,W=input_image.shape
    output_image=np.zeros([H,W],dtype=np.int32)

    padimage=np.pad(input_image,((H,H),(W,W)),'symmetric')
    padimage=np.array(padimage,dtype=np.int32)
    
    for i in range(H,2*H):
        for j in range(W,2*W):
            output_image[i-H,j-W]=stage_A(padimage,Smax,Sinit,i,j)

    a=np.max(output_image)
    b=np.min(output_image)

    for i in range(0,H):
        for j in range(0,W):
            output_image[i,j]=int((output_image[i,j]-b)*255/(a-b))
    
    output_image=np.array(output_image,dtype=np.uint8)
    return output_image
def stage_A(padimage,Smax,S,i,j):
    localimage=padimage[i-int((S-1)/2):i+int((S-1)/2)+1,j-int((S-1)/2):j+int((S-1)/2)+1]

    Z_min=np.min(localimage)
    Z_max=np.max(localimage)
    Z_med=np.median(localimage)

    A1=Z_med-Z_min
    A2=Z_med-Z_max

    if(A1>0 and A2<0):
        return stage_B(padimage[i,j],Z_min,Z_max,Z_med)
    else:
        S=S+2
        if(S<=Smax):
            return stage_A(padimage,Smax,S,i,j)
        else:
            return Z_med
def stage_B(Z_xy,Z_min,Z_max,Z_med):
    B1=Z_xy-Z_min
    B2=Z_xy-Z_max

    if(B1>0 and B2<0):
        return Z_xy
    else:
        return Z_med

def midpoint_filter(input_image,m,n):
    H,W=input_image.shape
    output_image=np.zeros([H,W],dtype=np.float32)
    padimage=np.pad(input_image,((H,H),(W,W)),'symmetric')
    padimage=np.array(padimage,dtype=np.float32)

    for i in range(H,2*H):
        for j in range(W,2*W):
            localimage=padimage[i-int((m-1)/2):i+int((m-1)/2)+1,j-int((n-1)/2):j+int((n-1)/2)+1]
            output_image[i-H,j-W]=(np.min(localimage)+np.max(localimage))/2

    a=np.max(output_image)
    b=np.min(output_image)
    for i in range(0,H):
        for j in range(0,W):
            output_image[i,j]=int((output_image[i,j]-b)*255/(a-b))
    
    output_image=np.array(output_image,dtype=np.uint8)
    return output_image

def Median_filter(input_image,size):
    H,W=input_image.shape
    output_image=np.zeros([H,W],dtype=np.int32)
    padimage=np.pad(input_image,((H,H),(W,W)),'symmetric')
    padimage=np.array(padimage,dtype=np.int32)

    for i in range(H,2*H):
        for j in range(W,2*W):
            localimage=padimage[i-int((size-1)/2):i+int((size-1)/2)+1,j-int((size-1)/2):j+int((size-1)/2)+1]
            output_image[i-H,j-W]=np.median(localimage)
    
    a=np.max(output_image)
    b=np.min(output_image)
    for i in range(0,H):
        for j in range(0,W):
            output_image[i,j]=int((output_image[i,j]-b)*255/(a-b))
    
    output_image=np.array(output_image,dtype=np.uint8)
    return output_image

def alpha_trimmed_mean_filter(input_image,m,n,d):
    H,W=input_image.shape
    output_image=np.zeros([H,W],dtype=np.float32)

    padimage=np.pad(input_image,((H,H),(W,W)),'symmetric')
    padimage=np.array(padimage,dtype=np.float32)

    for i in range(H,2*H):
        for j in range(W,W*2):
            local_image=padimage[i-int((m-1)/2):i+int((m-1)/2)+1,j-int((n-1)/2):j+int((n-1)/2)+1]
            order_local_image=np.sort(local_image.flatten())
            target_local=order_local_image[int(d/2):m*n-int(d/2)]
            output_image[i-H,j-W]=np.sum(target_local)/(m*n-d)
    
    a=np.max(output_image)
    b=np.min(output_image)
    for i in range(0,H):
        for j in range(0,W):
            output_image[i,j]=int((output_image[i,j]-b)*255/(a-b))
    
    output_image=np.array(output_image,dtype=np.uint8)
    return output_image


def Wiener_filter(input_image,K):
    input_image_DFT=np.fft.fft2(input_image)
    input_image_center=np.fft.fftshift(input_image_DFT)

    row,col=input_image_center.shape

    H=np.ones([row,col],dtype=np.complex128)
    for m in range (0,row):
        for n in range(0,col):
            if (m+n==(row+col)/2):
                H[m,n]=H[m,n]
            else:
                operator=np.pi*(0.1*(m-row/2)+0.1*(n-col/2))
                H[m,n]=1/operator*np.sin(operator)*np.exp(-1j*operator)
    
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


def constrained_least_squares_filter(input_image,gama):
    input_image_DFT=np.fft.fft2(input_image)
    input_image_center=np.fft.fftshift(input_image_DFT)


    row,col=input_image_center.shape

    H=np.ones([row,col],dtype=np.complex128)
    for m in range (0,row):
        for n in range(0,col):
            if (m+n==(row+col)/2):
                H[m,n]=H[m,n]
            else:
                operator=np.pi*(0.1*(m-row/2)+0.1*(n-col/2))
                H[m,n]=1/operator*np.sin(operator)*np.exp(-1j*operator)
    
    buf=np.conj(H)*H

    laplacian_operator=np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])
    pad_laplacian_operator=np.ones([row,col],dtype=np.complex128)
    pad_laplacian_operator[int((row-1-3)/2+1):int((row-1-3)/2+4),int((col-1-3)/2+1):int((col-1-3)/2+4)]=laplacian_operator

    laplacian_operator_DFT=np.fft.fft2(pad_laplacian_operator)
    laplacian_operator_center=np.fft.fftshift(laplacian_operator_DFT)

    laplacian_operator_power=np.conj(laplacian_operator_center)*laplacian_operator_center

    mask=np.conj(H)/(buf+gama*laplacian_operator_power)
    output_image=input_image_center*mask

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
    img=cv2.imread('Q6_3_1.tiff',cv2.IMREAD_GRAYSCALE)
    cv2.imshow('Q6_3_1',img)

    out1_1=Wiener_filter(img,0.1)
    cv2.imshow('Q6_3_1_restore_Wiener_filter_0.1',out1_1)
    savedimage1_1=Image.fromarray(out1_1)
    savedimage1_1.save('Q6_3_1_restore_Wiener_filter_0.1.tif')

    out1_2=Wiener_filter(img,0.01)
    cv2.imshow('Q6_3_1_restore_Wiener_filter_0.01',out1_2)
    savedimage1_2=Image.fromarray(out1_2)
    savedimage1_2.save('Q6_3_1_restore_Wiener_filter_0.01.tif')

    out1_3=Wiener_filter(img,0.001)
    cv2.imshow('Q6_3_1_restore_Wiener_filter_0.001',out1_3)
    savedimage1_3=Image.fromarray(out1_3)
    savedimage1_3.save('Q6_3_1_restore_Wiener_filter_0.001.tif')

    time_start = time.perf_counter()
    out1_4=Wiener_filter(img,0.0001)
    time_end = time.perf_counter()
    run_time = time_end - time_start
    print("运行时长：", run_time)
    cv2.imshow('Q6_3_1_restore_Wiener_filter_0.0001',out1_4)
    savedimage1_4=Image.fromarray(out1_4)
    savedimage1_4.save('Q6_3_1_restore_Wiener_filter_0.0001.tif')

    out1_5=Wiener_filter(img,0.00001)
    cv2.imshow('Q6_3_1_restore_Wiener_filter_0.00001',out1_5)
    savedimage1_5=Image.fromarray(out1_5)
    savedimage1_5.save('Q6_3_1_restore_Wiener_filter_0.00001.tif')

    out1_6=Wiener_filter(img,0.000001)
    cv2.imshow('Q6_3_1_restore_Wiener_filter_0.000001',out1_6)
    savedimage1_6=Image.fromarray(out1_6)
    savedimage1_6.save('Q6_3_1_restore_Wiener_filter_0.000001.tif')


    out2_1=constrained_least_squares_filter(img,0.1)
    cv2.imshow('Q6_3_1_restore_constrained_least_squares_filter_0.1',out2_1)
    savedimage2_1=Image.fromarray(out2_1)
    savedimage2_1.save('Q6_3_1_restore_constrained_least_squares_filter_0.1.tif')

    out2_2=constrained_least_squares_filter(img,0.01)
    cv2.imshow('Q6_3_1_restore_constrained_least_squares_filter_0.01',out2_2)
    savedimage2_2=Image.fromarray(out2_2)
    savedimage2_2.save('Q6_3_1_restore_constrained_least_squares_filter_0.01.tif')

    out2_3=constrained_least_squares_filter(img,0.001)
    cv2.imshow('Q6_3_1_restore_constrained_least_squares_filter_0.001',out2_3)
    savedimage2_3=Image.fromarray(out2_3)
    savedimage2_3.save('Q6_3_1_restore_constrained_least_squares_filter_0.001.tif')

    time_start2 = time.perf_counter()
    out2_4=Wiener_filter(img,0.0001)
    cv2.imshow('Q6_3_1_restore_constrained_least_squares_filter_0.0001',out2_4)
    time_end2 = time.perf_counter()
    run_time2 = time_end2 - time_start2
    savedimage2_4=Image.fromarray(out2_4)
    print("运行时长：", run_time2)
    savedimage2_4.save('Q6_3_1_restore_constrained_least_squares_filter_0.0001.tif')

    out2_5=constrained_least_squares_filter(img,0.00001)
    cv2.imshow('Q6_3_1_restore_constrained_least_squares_filter_0.00001',out2_5)
    savedimage2_5=Image.fromarray(out2_5)
    savedimage2_5.save('Q6_3_1_restore_constrained_least_squares_filter_0.00001.tif')

    out2_6=constrained_least_squares_filter(img,0.000001)
    cv2.imshow('Q6_3_1_restore_constrained_least_squares_filter_0.000001',out2_6)
    savedimage2_6=Image.fromarray(out2_6)
    savedimage2_6.save('Q6_3_1_restore_constrained_least_squares_filter_0.000001.tif')


    img2=cv2.imread('Q6_3_2.tiff',cv2.IMREAD_GRAYSCALE)
    cv2.imshow('Q6_3_2',img2)

    out3=Adaptive_Median_Filter(img2,7,3)
    cv2.imshow('Q6_3_2_processed_Adaptive_Median_filter',out3)
    savedimage3=Image.fromarray(out3)
    savedimage3.save('Q6_3_2_processed_Adaptive_Median_filter_1time.tif')


    out33=alpha_trimmed_mean_filter(img2,5,5,10)
    cv2.imshow('Q6_3_2_processed_alpha_trimmed_mean_filter',out33)
    savedimage33=Image.fromarray(out33)
    savedimage33.save('Q6_3_2_processed_alpha_trimmed_mean_filter.tif')

    out33_1=Wiener_filter(out33,0.1)
    cv2.imshow('Q6_3_2_restore_Wiener_filter_0.1',out33_1)
    savedimage33_1=Image.fromarray(out33_1)
    savedimage33_1.save('Q6_3_2_restore_Wiener_filter_0.1.tif')

    out33_2=Wiener_filter(out33,0.01)
    cv2.imshow('Q6_3_2_restore_Wiener_filter_0.01',out33_2)
    savedimage33_2=Image.fromarray(out33_2)
    savedimage33_2.save('Q6_3_2_restore_Wiener_filter_0.01.tif')

    out33_3=Wiener_filter(out33,0.001)
    cv2.imshow('Q6_3_2_restore_Wiener_filter_0.001',out33_3)
    savedimage33_3=Image.fromarray(out33_3)
    savedimage33_3.save('Q6_3_2_restore_Wiener_filter_0.001.tif')

    out33_4=Wiener_filter(out33,0.0001)
    cv2.imshow('Q6_3_2_restore_Wiener_filter_0.0001',out33_4)
    savedimage33_4=Image.fromarray(out33_4)
    savedimage33_4.save('Q6_3_2_restore_Wiener_filter_0.0001.tif')

    out33_5=Wiener_filter(out33,0.00001)
    cv2.imshow('Q6_3_2_restore_Wiener_filter_0.00001',out33_5)
    savedimage33_5=Image.fromarray(out33_5)
    savedimage33_5.save('Q6_3_2_restore_Wiener_filter_0.00001.tif')

    out33_6=Wiener_filter(out33,0.000001)
    cv2.imshow('Q6_3_2_restore_Wiener_filter_0.000001',out33_6)
    savedimage33_6=Image.fromarray(out33_6)
    savedimage33_6.save('Q6_3_2_restore_Wiener_filter_0.000001.tif')

    out34_1=constrained_least_squares_filter(out33,0.1)
    cv2.imshow('Q6_3_2_constrained_least_squares_filter_0.1',out34_1)
    savedimage34_1=Image.fromarray(out34_1)
    savedimage34_1.save('Q6_3_2_constrained_least_squares_filter_0.1.tif')

    out34_2=constrained_least_squares_filter(out33,0.01)
    cv2.imshow('Q6_3_2_constrained_least_squares_filter_0.01',out34_2)
    savedimage34_2=Image.fromarray(out34_2)
    savedimage34_2.save('Q6_3_2_constrained_least_squares_filter_0.01.tif')

    out34_3=constrained_least_squares_filter(out33,0.001)
    cv2.imshow('Q6_3_2_constrained_least_squares_filter_0.001',out34_3)
    savedimage34_3=Image.fromarray(out34_3)
    savedimage34_3.save('Q6_3_2_constrained_least_squares_filter_0.001.tif')

    out34_4=constrained_least_squares_filter(out33,0.0001)
    cv2.imshow('Q6_3_2_constrained_least_squares_filter_0.0001',out34_4)
    savedimage34_4=Image.fromarray(out34_4)
    savedimage34_4.save('Q6_3_2_constrained_least_squares_filter_0.0001.tif')

    out34_5=constrained_least_squares_filter(out33,0.00001)
    cv2.imshow('Q6_3_2_constrained_least_squares_filter_0.00001',out34_5)
    savedimage34_5=Image.fromarray(out34_5)
    savedimage34_5.save('Q6_3_2_constrained_least_squares_filter_0.00001.tif')

    out34_6=constrained_least_squares_filter(out33,0.000001)
    cv2.imshow('Q6_3_2_constrained_least_squares_filter_0.000001',out34_6)
    savedimage34_6=Image.fromarray(out34_6)
    savedimage34_6.save('Q6_3_2_constrained_least_squares_filter_0.000001.tif')
    



    img3=cv2.imread('Q6_3_3.tiff',cv2.IMREAD_GRAYSCALE)
    cv2.imshow('Q6_3_3',img3)

    out4=Adaptive_Median_Filter(img3,7,3)
    cv2.imshow('Q6_3_3_processed_Adaptive_Median_filter_1time',out4)
    savedimage3=Image.fromarray(out4)
    savedimage3.save('Q6_3_3_processed_Adaptive_Median_filter_1time.tif')

    out444=midpoint_filter(out4,3,3)
    out44=Median_filter(img3,3)
    out4444=alpha_trimmed_mean_filter(img3,5,5,10)
    
    cv2.imshow('Q6_3_3_processed_Median_filter',out44)
    cv2.imshow('Q6_3_3_processed_midpoint_filter_followed_adaptivemedianfilter',out444)
    cv2.imshow('Q6_3_3_processed_alpha_trimmed_mean_filter',out4444)


    savedimage44=Image.fromarray(out44)
    savedimage44.save('Q6_3_3_processed_Median_filter.tif')

    savedimage444=Image.fromarray(out444)
    savedimage444.save('Q6_3_3_processed_midpoint_filter_followed_adaptivemedianfilter.tif')

    savedimage4444=Image.fromarray(out4444)
    savedimage4444.save('Q6_3_3_processed_alpha_trimmed_mean_filter.tif')

    out44_1=Wiener_filter(out4444,0.1)
    cv2.imshow('Q6_3_3_restore_Wiener_filter_0.1',out44_1)
    savedimage44_1=Image.fromarray(out44_1)
    savedimage44_1.save('Q6_3_3_restore_Wiener_filter_0.1.tif')

    out44_2=Wiener_filter(out4444,0.01)
    cv2.imshow('Q6_3_3_restore_Wiener_filter_0.01',out44_2)
    savedimage44_2=Image.fromarray(out44_2)
    savedimage44_2.save('Q6_3_3_restore_Wiener_filter_0.01.tif')

    out44_3=Wiener_filter(out4444,0.001)
    cv2.imshow('Q6_3_3_restore_Wiener_filter_0.001',out44_3)
    savedimage44_3=Image.fromarray(out44_3)
    savedimage44_3.save('Q6_3_3_restore_Wiener_filter_0.001.tif')

    out44_4=Wiener_filter(out4444,0.0001)
    cv2.imshow('Q6_3_3_restore_Wiener_filter_0.0001',out44_4)
    savedimage44_4=Image.fromarray(out44_4)
    savedimage44_4.save('Q6_3_3_restore_Wiener_filter_0.0001.tif')

    out44_5=Wiener_filter(out4444,0.00001)
    cv2.imshow('Q6_3_3_restore_Wiener_filter_0.00001',out44_5)
    savedimage44_5=Image.fromarray(out44_5)
    savedimage44_5.save('Q6_3_3_restore_Wiener_filter_0.00001.tif')

    out44_6=Wiener_filter(out4444,0.000001)
    cv2.imshow('Q6_3_3_restore_Wiener_filter_0.000001',out44_6)
    savedimage44_6=Image.fromarray(out44_6)
    savedimage44_6.save('Q6_3_3_restore_Wiener_filter_0.000001.tif')

    out45_1=constrained_least_squares_filter(out4444,0.1)
    cv2.imshow('Q6_3_3_constrained_least_squares_filter_0.1',out45_1)
    savedimage45_1=Image.fromarray(out45_1)
    savedimage45_1.save('Q6_3_3_restore_constrained_least_squares_filter_0.1.tif')

    out45_2=constrained_least_squares_filter(out4444,0.01)
    cv2.imshow('Q6_3_3_constrained_least_squares_filter_0.01',out45_2)
    savedimage45_2=Image.fromarray(out45_2)
    savedimage45_2.save('Q6_3_3_restore_constrained_least_squares_filter_0.01.tif')

    out45_3=constrained_least_squares_filter(out4444,0.001)
    cv2.imshow('Q6_3_3_constrained_least_squares_filter_0.001',out45_3)
    savedimage45_3=Image.fromarray(out45_3)
    savedimage45_3.save('Q6_3_3_restore_constrained_least_squares_filter_0.001.tif')

    out45_4=constrained_least_squares_filter(out4444,0.0001)
    cv2.imshow('Q6_3_3_constrained_least_squares_filter_0.0001',out45_4)
    savedimage45_4=Image.fromarray(out45_4)
    savedimage45_4.save('Q6_3_3_restore_constrained_least_squares_filter_0.0001.tif')

    out45_5=constrained_least_squares_filter(out4444,0.00001)
    cv2.imshow('Q6_3_3_constrained_least_squares_filter_0.00001',out45_5)
    savedimage45_5=Image.fromarray(out45_5)
    savedimage45_5.save('Q6_3_3_restore_constrained_least_squares_filter_0.00001.tif')

    out45_6=constrained_least_squares_filter(out4444,0.000001)
    cv2.imshow('Q6_3_3_constrained_least_squares_filter_0.000001',out45_6)
    savedimage45_6=Image.fromarray(out45_6)
    savedimage45_6.save('Q6_3_3_restore_constrained_least_squares_filter_0.000001.tif')





    
    cv2.waitKey(0)

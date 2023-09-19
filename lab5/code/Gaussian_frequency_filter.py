import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math

def Gaussian_frequency_filter(input_image,D0):
    H,W=input_image.shape
    output_image_lowpass=np.zeros([H,W],dtype=np.float32)
    output_image_Highpass=np.zeros([H,W],dtype=np.float32)

    # pad input image and DFT
    P=2*H
    Q=2*W
    padimage=np.zeros([P,Q],dtype=np.float32)
    padimage[0:H,0:W]=input_image
    
    for i in range(0,P):
        for j in range(0,Q):
            padimage[i,j]=padimage[i,j]*((-1)**(i+j))
    
    padimage_DFT=np.fft.fft2(padimage)

    #Gaussian Lowpass Filter in frequency domain
    Gaussian_Lowpass=np.zeros([P,Q],dtype=np.complex128)
    for i in range(0,P):
        for j in range(0,Q):
            D=np.sqrt((i-P/2)**2+(j-Q/2)**2)
            Gaussian_Lowpass[i,j]=math.exp(-(D**2)/2/(D0**2))

    #Gaussian Highpass Filter in frequency domain
    Gaussian_Highpass=np.ones([P,Q],dtype=np.complex128)
    Gaussian_Highpass=Gaussian_Highpass-Gaussian_Lowpass



    #inverse of lowpass
    product_lowpass=padimage_DFT*Gaussian_Lowpass
    real_inverse_DFT_lowpass=np.real(np.fft.ifft2(product_lowpass))

    for i in range (0,real_inverse_DFT_lowpass.shape[0]):
        for j in range (0,real_inverse_DFT_lowpass.shape[1]):
            real_inverse_DFT_lowpass[i,j]=real_inverse_DFT_lowpass[i,j]*((-1)**(i+j))
    
    output_image_lowpass=real_inverse_DFT_lowpass[0:H,0:W]
    
    #a=np.max(output_image_lowpass)
    #b=np.min(output_image_lowpass)

    #for i in range(0,H):
    #    for j in range(0,W):
    #        output_image_lowpass[i,j]=int((output_image_lowpass[i,j]-b)*255/(a-b))

    #output_image_lowpass=np.array(output_image_lowpass,dtype=np.uint8)
    output_image_lowpass=np.clip(output_image_lowpass,0,255)
    output_image_lowpass=output_image_lowpass.astype(np.uint8)

    #inverse of highpass
    product_Highpass=padimage_DFT*Gaussian_Highpass
    real_inverse_DFT_Highpass=np.real(np.fft.ifft2(product_Highpass))

    for i in range (0,real_inverse_DFT_Highpass.shape[0]):
        for j in range (0,real_inverse_DFT_Highpass.shape[1]):
            real_inverse_DFT_Highpass[i,j]=real_inverse_DFT_Highpass[i,j]*((-1)**(i+j))
    
    output_image_Highpass=real_inverse_DFT_Highpass[0:H,0:W]
    
    #a=np.max(output_image_Highpass)
    #b=np.min(output_image_Highpass)

    #for i in range(0,H):
    #    for j in range(0,W):
    #        output_image_Highpass[i,j]=int((output_image_Highpass[i,j]-b)*255/(a-b))

    #output_image_Highpass=np.array(output_image_Highpass,dtype=np.uint8)
    output_image_Highpass=np.clip(output_image_Highpass, 0, 255)
    output_image_Highpass=output_image_Highpass.astype(np.uint8)

    return (output_image_lowpass,output_image_Highpass)

if __name__ == '__main__':
    img=cv2.imread('Q5_2.tif',cv2.IMREAD_GRAYSCALE)
    cv2.imshow('Q5_2',img)
    #cv2.waitKey(0)

    (out1,out2)=Gaussian_frequency_filter(img,30)

    cv2.imshow('Q5_2_Gaussian_Lowpass_D0=30',out1)
    #cv2.waitKey(0)
    savedimage1=Image.fromarray(out1)
    savedimage1.save('Q5_2_Gaussian_Lowpass_D0=30.tif')

    cv2.imshow('Q5_2_Gaussian_Highpass_D0=30',out2)
    cv2.waitKey(0)
    savedimage1=Image.fromarray(out2)
    savedimage1.save('Q5_2_Gaussian_Highpass_D0=30.tif')
    
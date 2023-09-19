import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def ILPF(input_image,D0):
    H,W=input_image.shape
    output_image=np.zeros([H,W],dtype=np.int32)

    # pad input image and DFT
    P=2*H
    Q=2*W
    padimage=np.zeros([P,Q],dtype=np.int32)
    padimage[0:H,0:W]=input_image

    for i in range (0,P):
        for j in range (0,Q):
            padimage[i,j]=padimage[i,j]*((-1)**(i+j))

    padimage_DFT=np.fft.fft2(padimage)
    #print(padimage_DFT)

    # ideal low pass filter in frequency domain
    ILPF_frequency=np.zeros([P,Q],dtype=np.complex128)
    for i in range(0,P):
        for j in range(0,Q):
            D=np.sqrt((i-P/2)**2+(j-Q/2)**2)
            if D<=D0:
                ILPF_frequency[i,j]=1
            else:
                ILPF_frequency[i,j]=0

    #inverse
    product=padimage_DFT*ILPF_frequency
    real_inverse_DFT=np.real(np.fft.ifft2(product))

    for i in range (0,real_inverse_DFT.shape[0]):
        for j in range (0,real_inverse_DFT.shape[1]):
            real_inverse_DFT[i,j]=real_inverse_DFT[i,j]*((-1)**(i+j))

    output_image=real_inverse_DFT[0:H,0:W]

    for i in range (0,output_image.shape[0]):
        for j in range (0,output_image.shape[1]):
            if output_image[i,j]<0:
                output_image[i,j]=0
            elif output_image[i,j]>255:
                output_image[i,j]=255


    #a=np.max(output_image)
    #b=np.min(output_image)

    #for i in range(0,H):
    #    for j in range(0,W):
    #        output_image[i,j]=int((output_image[i,j]-b)*255/(a-b))
    

    output_image=np.array(output_image,dtype=np.uint8)

    return output_image

if __name__ == '__main__':
    img=cv2.imread('Q5_2.tif',cv2.IMREAD_GRAYSCALE)
    cv2.imshow('Q5_2',img)
    #cv2.waitKey(0)

    out1=ILPF(img,10)
    cv2.imshow('Q5_2_D0=10',out1)
    cv2.waitKey(0)
    savedimage1=Image.fromarray(out1)
    savedimage1.save('Q5_2_D0=10.tif')


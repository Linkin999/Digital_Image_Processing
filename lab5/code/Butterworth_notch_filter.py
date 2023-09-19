import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math
import time

def DFT_input_image(input_image):
    H,W=input_image.shape

    #pad the input image
    P=2*H
    Q=2*W
    padimage=np.zeros([P,Q],dtype=np.float32)
    padimage[0:H,0:W]=input_image

    for i in range(0,P):
        for j in range(0,Q):
            padimage[i,j]=padimage[i,j]*((-1)**(i+j))

    padimage_DFT=np.fft.fft2(padimage)

    
    return padimage_DFT
    #return np.log(np.abs(padimage_DFT))


def Butterworth_notch_filter(input_image,coordinate,n,D0):
    H,W=input_image.shape

    P=2*H
    Q=2*W
    #Butterworth notch reject filter
    Butterworth_notch_reject_filter=np.ones([P,Q],dtype=np.float32)
    for i in range(0,P):
        for j in range(0,Q):
            for k in range(0,len(coordinate[0])):
                #D=np.sqrt((i-P/2-coordinate[0,k])**2+(j-Q/2-coordinate[1,k])**2)
                #D_inverse=np.sqrt((i-P/2+coordinate[0,k])**2+(j-Q/2+coordinate[1,k])**2)


                #D=np.sqrt((i-P/2-(coordinate[0,k]-P/2))**2+(j-Q/2-(coordinate[1,k]-Q/2))**2)
                #D_inverse=np.sqrt((i-P/2+(coordinate[0,k]-P/2))**2+(j-Q/2+(coordinate[1,k]-Q/2))**2)

                D=np.sqrt((i-coordinate[0,k])**2+(j-coordinate[1,k])**2)
                D_inverse=np.sqrt((i-(P/2-(coordinate[0,k]-P/2)))**2+(j-(Q/2-(coordinate[1,k]-Q/2)))**2)
                if D!=0 and D_inverse!=0:
                    Butterworth_notch_reject_filter[i,j]=Butterworth_notch_reject_filter[i,j]*(1/(1+(D0/D)**(2*n)))*(1/(1+(D0/D_inverse)**(2*n)))
                else:
                    Butterworth_notch_reject_filter[i,j]=0
    
    return Butterworth_notch_reject_filter

def obtain(DFT_input,DFT_filter,input_image):
    H,W=input_image.shape
    output_image=np.zeros([H,W],dtype=np.float32)
    product=DFT_input*DFT_filter
    real_inverse_DFT_lowpass=np.real(np.fft.ifft2(product))

    for i in range (0,real_inverse_DFT_lowpass.shape[0]):
        for j in range (0,real_inverse_DFT_lowpass.shape[1]):
            real_inverse_DFT_lowpass[i,j]=real_inverse_DFT_lowpass[i,j]*((-1)**(i+j))


    output_image=real_inverse_DFT_lowpass[0:H,0:W]
    output_image=np.clip(output_image,0,255)
    output_image=output_image.astype(np.uint8)

    return output_image
    







if __name__ == '__main__':
    img=cv2.imread('Q5_3.tif',cv2.IMREAD_GRAYSCALE)
    cv2.imshow('Q5_3',img)


    time_start = time.perf_counter()
    out1=DFT_input_image(img)
    #plt.imshow(np.log(np.abs(out1)),cmap=plt.cm.gray)

    #plt.show()

    
    coordinate=np.array([[87,79,169,161],[108,222,108,222]])
    n=4
    D0=20
    out2=Butterworth_notch_filter(img,coordinate,n,D0)
    #plt.imshow(np.abs(out2),cmap=plt.cm.gray)
    #plt.show()

    #plt.imshow(np.log(np.abs(out2*out1)),cmap=plt.cm.gray)
    #plt.show()

    out3=obtain(out1,out2,img)
    time_end = time.perf_counter()
    run_time = time_end - time_start
    print("运行时长：", run_time)

    cv2.imshow('Q5_3_Butterworth_notch_reject_filter',out3)
    cv2.waitKey(0)
    savedimage1=Image.fromarray(out3)
    savedimage1.save('Q5_3_D0=3.tif')




    





    




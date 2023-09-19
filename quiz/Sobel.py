import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time

def Sobel_spatial(input_image):
    H,W=input_image.shape
    output_image=np.zeros([H,W],dtype=np.int32)

    padimage=np.pad(input_image,((W,W),(H,H)),'symmetric')
    padimage=np.array(padimage,dtype=np.int32)
    
    for i in range(H,2*H):
        for j in range(W,2*W):
            #sobel_operator=padimage[i-1,j+1]+2*padimage[i,j+1]+padimage[i+1,j+1]-padimage[i-1,j-1]-2*padimage[i,j-1]-padimage[i+1,j-1]+padimage[i+1,j-1]+2*padimage[i+1,j]+padimage[i+1,j+1]-padimage[i-1,j-1]-2*padimage[i-1,j]-padimage[i-1,j+1]
            #output_image[i-H,j-W]=sobel_operator+128
            #sobel_operator=padimage[i-1,j+1]+2*padimage[i,j+1]+padimage[i+1,j+1]-padimage[i-1,j-1]-2*padimage[i,j-1]-padimage[i+1,j-1]
            #output_image[i-H,j-W]=sobel_operator+128
            sobel_operator=padimage[i+1,j-1]+2*padimage[i,j-1]+padimage[i-1,j-1]-padimage[i+1,j+1]-2*padimage[i,j+1]-padimage[i-1,j+1]
            output_image[i-H,j-W]=sobel_operator+128



    a=np.max(output_image)
    b=np.min(output_image)
    for i in range(0,H):
        for j in range(0,W):
            output_image[i,j]=int((output_image[i,j]-b)*255/(a-b))


    output_image=np.array(output_image,dtype=np.uint8)
    return output_image

def Sobel_frequency(input_image):
    H,W=input_image.shape
    output_image=np.zeros([H,W],dtype=np.int32)
    mask_H=3
    mask_W=3


    # pad input image and DFT
    P=H+mask_H-1
    Q=W+mask_W-1
    padimage=np.zeros([P,Q],dtype=np.int32)
    padimage[0:H,0:W]=input_image

    for i in range (0,P):
        for j in range (0,Q):
            padimage[i,j]=padimage[i,j]*((-1)**(i+j))

    padimage_DFT=np.fft.fft2(padimage)
    #print(padimage_DFT)

    #pad the mask and DFT
    sobel_operator=[[-1,0,1],[-2,0,2],[-1,0,1]]

    pad_sobel_operator=np.zeros([P,Q],dtype=np.int32)
    pad_sobel_operator[int(1+(P-1-mask_H)/2):int(1+(P-1-mask_H)/2+mask_H),int(1+(Q-1-mask_W)/2):int(1+(Q-1-mask_W)/2+mask_W)]=sobel_operator

    for i in range (0,P):
        for j in range (0,Q):
            pad_sobel_operator[i,j]=pad_sobel_operator[i,j]*((-1)**(i+j))
    
    pad_sobel_operator_DFT=np.fft.fft2(pad_sobel_operator)
    #print("hello")
    #print(pad_sobel_operator_DFT)
    pad_sobel_operator_DFT.real=0
    

    for u in range (0,pad_sobel_operator_DFT.shape[0]):
        for v in range (0,pad_sobel_operator_DFT.shape[1]):
            pad_sobel_operator_DFT[u,v]=pad_sobel_operator_DFT[u,v]*((-1)**(u+v))

    #print(pad_sobel_operator_DFT)


    #inverse
    product=padimage_DFT*pad_sobel_operator_DFT
    real_inverse_DFT=np.real(np.fft.ifft2(product))

    for i in range (0,real_inverse_DFT.shape[0]):
        for j in range (0,real_inverse_DFT.shape[1]):
            real_inverse_DFT[i,j]=real_inverse_DFT[i,j]*((-1)**(i+j))

    output_image=real_inverse_DFT[0:H,0:W]

    a=np.max(output_image)
    b=np.min(output_image)
    for i in range(0,H):
        for j in range(0,W):
            output_image[i,j]=int((output_image[i,j]-b)*255/(a-b))

    output_image=np.array(output_image,dtype=np.uint8)
    return output_image


if __name__ == '__main__':
    #img=cv2.imread('21.jpg',cv2.IMREAD_GRAYSCALE)
    #cv2.imshow('21.jpg',img)

    img1=cv2.imread('1.jpg',cv2.IMREAD_GRAYSCALE)
    img2=cv2.imread('2.jpg',cv2.IMREAD_GRAYSCALE)
    img3=cv2.imread('3.jpg',cv2.IMREAD_GRAYSCALE)
    img4=cv2.imread('4.jpg',cv2.IMREAD_GRAYSCALE)
    img5=cv2.imread('5.jpg',cv2.IMREAD_GRAYSCALE)
    img6=cv2.imread('6.jpg',cv2.IMREAD_GRAYSCALE)
    img7=cv2.imread('7.jpg',cv2.IMREAD_GRAYSCALE)
    img8=cv2.imread('8.jpg',cv2.IMREAD_GRAYSCALE)
    img9=cv2.imread('9.jpg',cv2.IMREAD_GRAYSCALE)
    img10=cv2.imread('10.jpg',cv2.IMREAD_GRAYSCALE)
    img11=cv2.imread('11.jpg',cv2.IMREAD_GRAYSCALE)
    img12=cv2.imread('12.jpg',cv2.IMREAD_GRAYSCALE)
    img13=cv2.imread('13.jpg',cv2.IMREAD_GRAYSCALE)
    img14=cv2.imread('14.jpg',cv2.IMREAD_GRAYSCALE)
    img15=cv2.imread('15.jpg',cv2.IMREAD_GRAYSCALE)
    img16=cv2.imread('16.jpg',cv2.IMREAD_GRAYSCALE)
    img17=cv2.imread('17.jpg',cv2.IMREAD_GRAYSCALE)
    img18=cv2.imread('18.jpg',cv2.IMREAD_GRAYSCALE)
    img19=cv2.imread('19.jpg',cv2.IMREAD_GRAYSCALE)
    img20=cv2.imread('20.jpg',cv2.IMREAD_GRAYSCALE)
    img21=cv2.imread('21.jpg',cv2.IMREAD_GRAYSCALE)
    img22=cv2.imread('22.jpg',cv2.IMREAD_GRAYSCALE)
    img23=cv2.imread('23.jpg',cv2.IMREAD_GRAYSCALE)
    img24=cv2.imread('24.jpg',cv2.IMREAD_GRAYSCALE)
    img25=cv2.imread('25.jpg',cv2.IMREAD_GRAYSCALE)

    img=(img1+img2+img3+img4+img5+img6+img7+img8+img9+img10+img11+img12+img13+img14+img15+img16+img17+img18+img19+img20+img21+img22+img23+img24+img25)/25
    a=np.max(img)
    b=np.min(img)
    for i in range(0,img.shape[0]):
        for j in range(0,img.shape[1]):
            img[i,j]=int((img[i,j]-b)*255/(a-b))

    img=np.array(img,dtype=np.uint8)

    img_1=img1-img
    img_1=np.clip(img_1, 0, 255)

    img_2=img2-img
    img_2=np.clip(img_2, 0, 255)

    img_3=img3-img
    img_3=np.clip(img_3, 0, 255)

    img_4=img4-img
    img_4=np.clip(img_4, 0, 255)

    img_5=img5-img
    img_5=np.clip(img_5, 0, 255)

    img_6=img6-img
    img_6=np.clip(img_6, 0, 255)

    img_7=img7-img
    img_7=np.clip(img_7, 0, 255)

    img_8=img8-img
    img_8=np.clip(img_8, 0, 255)

    img_9=img9-img
    img_9=np.clip(img_9, 0, 255)

    img_10=img10-img
    img_10=np.clip(img_10, 0, 255)

    img_11=img11-img
    img_11=np.clip(img_11, 0, 255)

    img_12=img12-img
    img_12=np.clip(img_12, 0, 255)

    img_13=img13-img
    img_13=np.clip(img_13, 0, 255)

    img_14=img14-img
    img_14=np.clip(img_14, 0, 255)

    img_15=img15-img
    img_15=np.clip(img_15, 0, 255)

    img_16=img16-img
    img_16=np.clip(img_16, 0, 255)

    img_17=img17-img
    img_17=np.clip(img_17, 0, 255)

    img_18=img18-img
    img_18=np.clip(img_18, 0, 255)

    img_19=img19-img
    img_19=np.clip(img_19, 0, 255)

    img_20=img20-img
    img_20=np.clip(img_20, 0, 255)

    img_21=img21-img
    img_21=np.clip(img_21, 0, 255)

    img_22=img22-img
    img_22=np.clip(img_22, 0, 255)

    img_23=img23-img
    img_23=np.clip(img_23, 0, 255)

    img_24=img24-img
    img_24=np.clip(img_24, 0, 255)

    img_25=img25-img
    img_25=np.clip(img_25, 0, 255)
    #cv2.waitKey(0)

    #time_start = time.perf_counter()
    out1=Sobel_spatial(img_1)
    #time_end = time.perf_counter()
    #run_time = time_end - time_start
    #print("运行时长：", run_time)
    cv2.imshow('21_spatial.jpg',out1)
    #cv2.waitKey(0)
    #savedimage1=Image.fromarray(out1)
    #savedimage1.save('Q5_1_Sobel_spatial.tif')


    #time_start2 = time.perf_counter()
    out2=Sobel_frequency(img_1)
    #time_end2 = time.perf_counter()
    #run_time2 = time_end2 - time_start2
    #print("运行时长：", run_time2)
    #plt.imshow(out2,cmap=plt.cm.gray)
    #plt.show()
    cv2.imshow('21_frequency.jpg',out2)
    cv2.waitKey(0)
    #savedimage2=Image.fromarray(out2)
    #savedimage2.save('Q5_1_Sobel_frequency.tif')


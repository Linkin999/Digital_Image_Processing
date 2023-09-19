import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math
import time

def hist_equ_12011923(input_image):
    H,W=input_image.shape
    total=H*W*1
    
    output_image=np.zeros([H,W],dtype=np.uint8)
    input_hist=[]
    output_hist=[]
    
    #input_histogram
    for i in range(256):
        input_hist.append(np.sum(input_image==i)/(total))
        
        
    sum_h=0
    for i in range(0,256):
        ind=np.where(input_image==i)
        sum_h+=len(input_image[ind])
        z_prime=round(255/total*sum_h)
        output_image[ind]=z_prime
        
        
    #output_histogram
    for i in range(256):
        output_hist.append(np.sum(output_image==i)/(total))
        
    return (output_image,input_hist,output_hist)

def Laplace_operator(input_image):
    H,W=input_image.shape
    output_image=np.zeros([H,W])
    
    #pad the input_image
    padimage=np.pad(input_image,((W,W),(H,H)),'symmetric')
    padimage=np.array(padimage, dtype=np.int32)
    #print(padimage.shape)
    #print(padimage.dtype)
    for i in range(H,2*H):
        for j in range(W,2*W):
            operator=padimage[i+1,j]+padimage[i-1,j]+padimage[i,j-1]+padimage[i,j+1]-4*padimage[i,j]
            output_image[i-H,j-W]=padimage[i,j]-operator

    #a=np.max(output_image)
    #b=np.min(output_image)
    #for i in range(0,H):
    #    for j in range(0,W):
    #        output_image[i,j]=int((output_image[i,j]-b)*255/(a-b))


    #output_image=np.array(output_image,dtype=np.uint8)
    return output_image

def local_hist_equ_12011923(input_image,m_size):
    H,W=input_image.shape
    total=H*W*1

    output_image=np.zeros([H,W],dtype=np.uint8)
    input_hist=[]
    output_hist=[]
    
    #input_histogram
    for i in range(256):
        input_hist.append(np.sum(input_image==i)/(total))
        
    #往四周补0
    padimage=np.pad(input_image,((int((m_size-1)/2),int((m_size-1)/2)),(int((m_size-1)/2),int((m_size-1)/2))),'constant',constant_values=(0,0))
    #local histogram processing
    for i in range(int((m_size-1)/2),input_image.shape[0]+int((m_size-1)/2)):
        for j in range(int((m_size-1)/2),input_image.shape[1]+int((m_size-1)/2)):
            partimage=padimage[i-int((m_size-1)/2):i+int((m_size-1)/2)+1,j-int((m_size-1)/2):j+int((m_size-1)/2)+1]
            part_hist=np.zeros(256)
            for m in range(partimage.shape[0]):
                for n in range(partimage.shape[1]):
                    part_hist[partimage[m][n]]=part_hist[partimage[m][n]]+1
            
            r_s=np.cumsum(part_hist[:partimage[int((m_size-1)/2),int((m_size-1)/2)]+1])/(m_size*m_size)
            output_image[i-int((m_size-1)/2),j-int((m_size-1)/2)]=int(round(255*r_s[partimage[int((m_size-1)/2),int((m_size-1)/2)]]))
            
        
    #output_histogram
    for i in range(256):
        output_hist.append(np.sum(output_image==i)/(total))
    
    return (output_image,output_hist,input_hist)

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


if __name__=='__main__':
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

    #img1=img1.astype(np.float32)
    #img2=img2.astype(np.float32)
    #img3=img3.astype(np.float32)
    #img4=img4.astype(np.float32)
    #img5=img5.astype(np.float32)
    #img6=img6.astype(np.float32)
    #img7=img7.astype(np.float32)
    #img8=img8.astype(np.float32)
    #img9=img9.astype(np.float32)
    #img10=img10.astype(np.float32)
    #img11=img11.astype(np.float32)
    #img12=img12.astype(np.float32)
    #img13=img13.astype(np.float32)
    #img14=img14.astype(np.float32)
    #img15=img15.astype(np.float32)
    #img16=img16.astype(np.float32)
    #img17=img17.astype(np.float32)
    #img18=img18.astype(np.float32)
    #img19=img19.astype(np.float32)
    #img20=img20.astype(np.float32)
    #img21=img21.astype(np.float32)
    #img22=img22.astype(np.float32)
    #img23=img23.astype(np.float32)
    #img4=img4.astype(np.float32)
    #img5=img5.astype(np.float32)
    #img6=img6.astype(np.float32)
    #img7=img7.astype(np.float32)
    #img8=img8.astype(np.float32)
    #img9=img9.astype(np.float32)
    #img10=img10.astype(np.float32)

    
    


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






    #cv2.imshow('average.jpg',img)
    #cv2.imshow('substract1.jpg',img_1)
    savedimage1=Image.fromarray(img_25)
    savedimage1.save('substracted 25.jpg')
    #cv2.imshow('substract2.jpg',img_2)
    #cv2.imshow('substract3.jpg',img_3)

    #cv2.imshow('substract4.jpg',img_4)
    #cv2.imshow('substract5.jpg',img_5)

    #cv2.imshow('substract6.jpg',img_6)
    #cv2.imshow('substract7.jpg',img_7)
    #cv2.imshow('substract8.jpg',img_8)
    #cv2.imshow('substract9.jpg',img_9)
    #cv2.imshow('substract10.jpg',img_10)
    #cv2.imshow('substract11.jpg',img_11)
    #cv2.imshow('substract12.jpg',img_12)
    #cv2.imshow('substract13.jpg',img_13)

    #cv2.imshow('substract14.jpg',img_14)
    #cv2.imshow('substract15.jpg',img_15)

    #cv2.imshow('substract16.jpg',img_16)
    #cv2.imshow('substract17.jpg',img_17)
    #cv2.imshow('substract18.jpg',img_18)
    #cv2.imshow('substract19.jpg',img_19)
    #cv2.imshow('substract20.jpg',img_20)

    #cv2.imshow('substract21.jpg',img_21)
    #cv2.imshow('substract22.jpg',img_22)
    #cv2.imshow('substract23.jpg',img_23)

    #cv2.imshow('substract24.jpg',img_24)
    #cv2.imshow('substract25.jpg',img_25)
    #cv2.waitKey(0)

    out=Laplace_operator(img_25)

    #cv2.imshow('processed by Laplace_operator.jpg',out)
    cv2.imwrite('substracted 25 processed by Laplace_operator.jpg',out)
    
    #cv2.waitKey(0)

    (out1,out2)=Gaussian_frequency_filter(img_25,30)
    #cv2.imshow('lowpass.jpg',out1)
    savedimage3=Image.fromarray(out1)
    savedimage3.save('substracted 25 processed by Gaussian_lowpass_filte.jpg')
    #cv2.imshow('highpass.jpg',out2)
    savedimage4=Image.fromarray(out2)
    savedimage4.save('substracted 25 processed by Gaussian_high_filte.jpg')

    (output_image,input_hist,output_hist)=hist_equ_12011923(img_25)
    #cv2.imshow('processed by histogram equalization.jpg',output_image)
    savedimage5=Image.fromarray(output_image)
    savedimage5.save('substracted 25 processed by histogram equalization.jpg')
    #cv2.waitKey(0)

    (output_image1,output_hist,input_hist)=local_hist_equ_12011923(img_25,3)
    #cv2.imshow('processed by local histogram equalization.jpg',output_image1)
    savedimage6=Image.fromarray(output_image1)
    savedimage6.save('substracted 25 processed by local histogram equalization.jpg')
    #cv2.waitKey(0)

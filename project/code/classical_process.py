import time
import numpy as np
import cv2
def sharpened_convolution(src):#进行锐化卷积
    H,W=src.shape
    output_image=np.zeros([H,W],dtype=np.int32)

    #锐化卷积算子
    kernel=np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])

    #pad src
    padimage=np.pad(src,((1,1),(1,1)),'edge')

    for i in range(1,H+1):
        for j in range(1,W+1):
            output_image[i-1][j-1]=np.sum(padimage[i-1:i+2,j-1:j+2]*kernel)
    
    #normalize
    a=np.max(output_image)
    b=np.min(output_image)
    
    for i in range(0,output_image.shape[0]):
        for j in range(0,output_image.shape[1]):
            output_image[i][j]=int((output_image[i][j]-b)/(a-b)*255)
    
    output_image=np.array(output_image,dtype=np.uint8)

    return output_image

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


def Perceptual_hash(image):
    #缩放图像 32×32
    resizedImage=cv2.resize(image,(32,32),interpolation=cv2.INTER_CUBIC)
    resizedImage=resizedImage.astype(np.float32)

    #DCT变换
    DCT_image=cv2.dct(resizedImage)

    #取左上角8×8
    DCT_image2=DCT_image[0:8,0:8]

    #计算平均值
    average=np.sum(DCT_image2)/64

    #计算哈希值
    hash_str=''
    for i in range(0,8):
        for j in range(0,8):
            if(DCT_image2[i][j]<average):
                hash_str=hash_str+'0'
            else:
                hash_str=hash_str+'1'
    return hash_str

#计算汉明距离
def hamming_distance(hash1,hash2):
    count=0
    for i in range(0,len(hash1)):
        if(hash1[i]==hash2[i]):
            count=count+1
    return count

#判断两者是否是同一个
"""def similarity(count):
    ratio=count/64
    if(ratio>0.92):"""
        

if  __name__=='__main__':
    """a=np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
    b=np.array([[1,1,1],[1,1,1],[1,1,1]])
    print(np.sum(a*b))"""

    """img=cv2.imread('Q4_2.tif',cv2.IMREAD_GRAYSCALE)
    cv2.imshow('original',img)

    out=sharpened_convolution(img)
    cv2.imshow('processed',out)"""

    template=cv2.imread('BioID_0107.pgm',cv2.IMREAD_GRAYSCALE)
    print(template.shape)
    cv2.imshow('template',template)
    #template1=Sobel_spatial(template)
    time_start_template = time.perf_counter()
    template_hash=Perceptual_hash(template)
    time_end_template = time.perf_counter()
    run_time_template= time_end_template - time_start_template
    print("运行时长：", run_time_template)

    test1=cv2.imread('BioID_0100.pgm',cv2.IMREAD_GRAYSCALE)
    cv2.imshow('test1',test1)
    #test1_sharpened=Sobel_spatial(test1)
    time_start_test1 = time.perf_counter()
    test1_hash=Perceptual_hash(test1)
    time_end_test1 = time.perf_counter()
    run_time_test1= time_end_test1 - time_start_test1
    print("运行时长：", run_time_test1)

    test2=cv2.imread('BioID_0118.pgm',cv2.IMREAD_GRAYSCALE)
    cv2.imshow('test2',test2)
    #test2_sharpened=Sobel_spatial(test2)
    time_start_test2 = time.perf_counter()
    test2_hash=Perceptual_hash(test2)
    time_end_test2 = time.perf_counter()
    run_time_test2= time_end_test2 - time_start_test2
    print("运行时长：", run_time_test2)

    print(template_hash)
    print("\n")
    print(test1_hash)
    print("\n")
    print(test2_hash)
    print("\n")

    template_test1=hamming_distance(template_hash,test1_hash)
    template_test2=hamming_distance(template_hash,test2_hash)
    test2_test1=hamming_distance(test2_hash,test1_hash)

    print(1-template_test1/64)
    print("\n")
    print(1-template_test2/64)
    print("\n")
    print(1-test2_test1/64)
    print("\n")


    cv2.waitKey(0)
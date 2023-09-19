from PIL import Image
import numpy as np
import cv2
import time

def contraharmonic_mean_filter(input_image,size,order):#size should be odd number
    H,W=input_image.shape
    output_image=np.zeros([H,W],dtype=np.float32)

    padimage=np.pad(input_image,((H,H),(W,W)),'symmetric')
    padimage=np.array(padimage,dtype=np.float32)

    for i in range(H,2*H):
        for j in range(W,2*W):
            operator1=0
            operator2=0
            for m in range(int(i-(size-1)/2),1+int(i+(size-1)/2)):
                for n in range(int(j-(size-1)/2),1+int(j+(size-1)/2)):
                    if(padimage[m,n]==0):
                        operator1=operator1+1
                        operator2=operator2+1
                    else:
                        operator1=operator1+padimage[m,n]**(order+1)
                        operator2=operator2+padimage[m,n]**(order)
            if(operator2==0):
                output_image[i-H,j-W]=0
            else:
                output_image[i-H,j-W]=operator1/operator2

    a=np.max(output_image)
    b=np.min(output_image)

    for i in range(0,H):
        for j in range(0,W):
            output_image[i,j]=int((output_image[i,j]-b)*255/(a-b))
    
    output_image=np.array(output_image,dtype=np.uint8)
    return output_image

def max_filter(input_image,size):
    H,W=input_image.shape
    output_image=np.zeros([H,W],dtype=np.int32)
    padimage=np.pad(input_image,((H,H),(W,W)),'symmetric')
    padimage=np.array(padimage,dtype=np.int32)

    for i in range(H,2*H):
        for j in range(W,2*W):
            localimage=padimage[i-int((size-1)/2):i+int((size-1)/2)+1,j-int((size-1)/2):j+int((size-1)/2)+1]
            output_image[i-H,j-W]=np.max(localimage)
    
    a=np.max(output_image)
    b=np.min(output_image)
    for i in range(0,H):
        for j in range(0,W):
            output_image[i,j]=int((output_image[i,j]-b)*255/(a-b))
    
    output_image=np.array(output_image,dtype=np.uint8)
    return output_image

def min_filter(input_image,size):
    H,W=input_image.shape
    output_image=np.zeros([H,W],dtype=np.int32)
    padimage=np.pad(input_image,((H,H),(W,W)),'symmetric')
    padimage=np.array(padimage,dtype=np.int32)

    for i in range(H,2*H):
        for j in range(W,2*W):
            localimage=padimage[i-int((size-1)/2):i+int((size-1)/2)+1,j-int((size-1)/2):j+int((size-1)/2)+1]
            output_image[i-H,j-W]=np.min(localimage)
    
    a=np.max(output_image)
    b=np.min(output_image)
    for i in range(0,H):
        for j in range(0,W):
            output_image[i,j]=int((output_image[i,j]-b)*255/(a-b))
    
    output_image=np.array(output_image,dtype=np.uint8)
    return output_image

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

def Arithmetic_mean_filter(input_image,m,n):
    H,W=input_image.shape
    output_image=np.zeros([H,W],dtype=np.float32)

    padimage=np.pad(input_image,((H,H),(W,W)),'symmetric')
    padimage=np.array(padimage,dtype=np.float32)

    for i in range(H,2*H):
        for j in range(W,2*W):
            local_image=padimage[i-int((m-1)/2):i+int((m-1)/2)+1,j-int((n-1)/2):j+int((n-1)/2)+1]
            output_image[i-H,j-W]=np.sum(local_image)/(m*n)
    
    a=np.max(output_image)
    b=np.min(output_image)
    for i in range(0,H):
        for j in range(0,W):
            output_image[i,j]=int((output_image[i,j]-b)*255/(a-b))
    
    output_image=np.array(output_image,dtype=np.uint8)
    return output_image

def Geometric_mean_filter(input_image,m,n):
    H,W=input_image.shape
    output_image=np.zeros([H,W],dtype=np.float64)
    padimage=np.pad(input_image,((H,H),(W,W)),'symmetric')
    padimage=np.array(padimage,dtype=np.float64)

    for i in range(H,2*H):
        for j in range(W,2*W):
            local_image=padimage[i-int((m-1)/2):i+int((m-1)/2)+1,j-int((n-1)/2):j+int((n-1)/2)+1]
            tmp=np.prod(local_image)
            output_image[i-H,j-W]=np.power(tmp,1/(m*n))

    print(output_image)
    a=np.max(output_image)
    b=np.min(output_image)
    for i in range(0,H):
        for j in range(0,W):
            output_image[i,j]=int((output_image[i,j]-b)*255/(a-b))
    
    print(output_image)
    output_image=np.array(output_image,dtype=np.uint8)
    print(output_image)
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


if __name__=='__main__':
    img=cv2.imread('Q6_1_1.tiff',cv2.IMREAD_GRAYSCALE)
    cv2.imshow('Q6_1_1',img)

    time_start = time.perf_counter()
    out=contraharmonic_mean_filter(img,3,1.5)
    time_end = time.perf_counter()
    run_time = time_end - time_start
    print("运行时长：", run_time)
    cv2.imshow('Q6_1_1_processed_contraharmonic_mean_filter_1.5',out)
    savedimage1=Image.fromarray(out)
    savedimage1.save('Q6_1_1_processed_contraharmonic_mean_filter_1.5.tif')

    time_start11 = time.perf_counter()
    out11=max_filter(img,3)
    time_end11 = time.perf_counter()
    run_time11 = time_end11 - time_start11
    print("运行时长：", run_time11)
    cv2.imshow('Q6_1_1_processed_max_filter',out11)
    savedimage11=Image.fromarray(out11)
    savedimage11.save('Q6_1_1_processed_max_filter.tif')

    

    img2=cv2.imread('Q6_1_2.tiff',cv2.IMREAD_GRAYSCALE)
    cv2.imshow('Q6_1_2',img2)

    time_start2 = time.perf_counter()
    out2=contraharmonic_mean_filter(img2,3,-1.5)
    time_end2 = time.perf_counter()
    run_time2 = time_end2- time_start2
    print("运行时长：", run_time2)
    cv2.imshow('Q6_1_2_processed_contraharmonic_mean_filter_(-1.5)',out2)
    savedimage2=Image.fromarray(out2)
    savedimage2.save('Q6_1_2_processed_contraharmonic_mean_filter_(-1.5).tif')

    time_start22 = time.perf_counter()
    out22=min_filter(img2,3)
    time_end22 = time.perf_counter()
    run_time22= time_end22- time_start22
    print("运行时长：", run_time22)
    cv2.imshow('Q6_1_2_processed_min_filter',out22)
    savedimage22=Image.fromarray(out22)
    savedimage22.save('Q6_1_2_processed_min_filter.tif')

    img3=cv2.imread('Q6_1_3.tiff',cv2.IMREAD_GRAYSCALE)
    cv2.imshow('Q6_1_3',img3)

    time_start3 = time.perf_counter()
    out3=Adaptive_Median_Filter(img3,7,3)
    time_end3 = time.perf_counter()
    run_time3= time_end3- time_start3
    print("运行时长：", run_time3)
    cv2.imshow('Q6_1_3_processed_Adaptive_Median_filter',out3)
    savedimage3=Image.fromarray(out3)
    savedimage3.save('Q6_1_3_processed_Adaptive_Median_filter.tif')

    time_start4 = time.perf_counter()
    out4=Median_filter(img3,7)
    time_end4 = time.perf_counter()
    run_time4= time_end4- time_start4
    print("运行时长：", run_time4)
    cv2.imshow('Q6_1_3_processed_Median_filter',out4)
    savedimage4=Image.fromarray(out4)
    savedimage4.save('Q6_1_3_processed_Median_filter.tif')

    img4=cv2.imread('Q6_1_4.tiff',cv2.IMREAD_GRAYSCALE)
    cv2.imshow('Q6_1_4',img4)

    out5=Arithmetic_mean_filter(img4,5,5)
    cv2.imshow('Q6_1_4_processed_arithmetric_mean_filter',out5)

    out6=Geometric_mean_filter(img4,5,5)
    cv2.imshow('Q6_1_4_processed_geometric_mean_filter',out6)

    time_start7 = time.perf_counter()
    out7=alpha_trimmed_mean_filter(img4,5,5,10)
    time_end7 = time.perf_counter()
    run_time7= time_end7- time_start7
    print("运行时长：", run_time7)
    cv2.imshow('Q6_1_4_processed_alpha_trimmed_mean_filter',out7)
    savedimage7=Image.fromarray(out7)
    savedimage7.save('Q6_1_4_processed_alpha_trimmed_mean_filter_5×5_d=10.tif')
    
    out8=Median_filter(img4,3)
    cv2.imshow('Q6_1_4_processed_median_filter3',out8)
    savedimage8=Image.fromarray(out8)
    savedimage8.save('Q6_1_4_processed_Median_filter_3×3.tif')
    out88=Median_filter(img4,5)
    cv2.imshow('Q6_1_4_processed_median_filter5',out88)
    savedimage88=Image.fromarray(out88)
    savedimage88.save('Q6_1_4_processed_Median_filter_5×5.tif')
    out888=Median_filter(img4,7)
    cv2.imshow('Q6_1_4_processed_median_filter7',out888)
    savedimage888=Image.fromarray(out888)
    savedimage888.save('Q6_1_4_processed_Median_filter_7×7.tif')

    out9=Adaptive_Median_Filter(img4,7,5)
    cv2.imshow('Q6_1_4_processed_adaptive_median_filter',out9)
    savedimage9=Image.fromarray(out9)
    savedimage9.save('Q6_1_4_processed_daptive_median_filter_max7.tif')
    

    time_start10 = time.perf_counter()
    out10=midpoint_filter(Adaptive_Median_Filter(img4,7,5),3,3)
    time_end10 = time.perf_counter()
    run_time10= time_end10- time_start10
    print("运行时长：", run_time10)
    cv2.imshow('Q6_1_4_processed_midpoint_filter_followedByAdaptiveMeanFilter',out10)
    savedimage10=Image.fromarray(out10)
    savedimage10.save('Q6_1_4_processed_midpoint_filter_3×3_followedByAdaptiveMeanFilter.tif')

    time_start11 = time.perf_counter()
    out11=midpoint_filter(Median_filter(img4,5),3,3)
    time_end11 = time.perf_counter()
    run_time11= time_end11- time_start11
    print("运行时长：", run_time11)
    cv2.imshow('Q6_1_4_processed_midpoint_filter_followedByMedianFilter',out11)
    savedimage11=Image.fromarray(out11)
    savedimage11.save('Q6_1_4_processed_midpoint_filter_3×3_followedByMedianFilter.tif')


    cv2.waitKey(0)



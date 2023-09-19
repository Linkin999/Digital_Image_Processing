import math
import time
import numpy as np
import cv2
import skimage.feature
import matplotlib.pyplot as plt

##---------------------##
#以下是关于原始LBP的计算
##---------------------##

def LBP_original(image):
    original_array=np.zeros(image.shape,dtype=np.uint8)
    H,W=image.shape

    #pad image,cv2.resize是先宽度再高度
    pad_image=cv2.resize(image,(W+2,H+2),interpolation=cv2.INTER_CUBIC)

    #计算每个像素点的LBP值
    for i in range(1,H+1):
        for j in range(1,W+1):
            original_array[i-1,j-1]=bit2decimal(calculate_original(pad_image,i,j))

    return original_array

#计算每个像素点领域所构成的比特序列
def calculate_original(image,i,j):
    bitSequence=[]
    if(image[i-1][j-1]>image[i][j]):
        bitSequence.append(1)
    else:
        bitSequence.append(0)
    
    if(image[i-1][j]>image[i][j]):
        bitSequence.append(1)
    else:
        bitSequence.append(0)
    
    if(image[i-1][j+1]>image[i][j]):
        bitSequence.append(1)
    else:
        bitSequence.append(0)

    if(image[i][j-1]>image[i][j]):
        bitSequence.append(1)
    else:
        bitSequence.append(0)

    if(image[i][j+1]>image[i][j]):
        bitSequence.append(1)
    else:
        bitSequence.append(0)

    if(image[i+1][j-1]>image[i][j]):
        bitSequence.append(1)
    else:
        bitSequence.append(0)
    
    if(image[i+1][j]>image[i][j]):
        bitSequence.append(1)
    else:
        bitSequence.append(0)

    if(image[i+1][j+1]>image[i][j]):
        bitSequence.append(1)
    else:
        bitSequence.append(0)
    return bitSequence

#计算比特序列对应的十进制值
def bit2decimal(bitSequence):
    decimal=0
    #print(bitSequence.shape[1])
    for i in range(0,8):
        decimal=decimal+bitSequence[i]*np.power(2,7-i)
    #print(decimal)
    return decimal


def calculate_histogram_original(original_array):
    H,W=original_array.shape
    total=H*W*1
    histogram=[]
    for i in range(0,256):
        histogram.append(np.sum(original_array==i)/total)

    return histogram


##---------------------##
#以下是关于旋转不变LBP的计算
##---------------------##
# P是采样点，R是半径
def LBP_Rotation_invariant(image,P,R):
    Rotation_invariant_array=np.zeros(image.shape,dtype=np.uint8)
    H,W=image.shape

    #pad image,cv2.resize是先宽度再高度
    pad_image=cv2.resize(image,(W+2*R,H+2*R),interpolation=cv2.INTER_CUBIC)

    #计算每个像素点的LBP值
    for i in range(R,H+R):
        for j in range(R,W+R):
            Rotation_invariant_array[i-R][j-R]=findMinDecimal_Rotation_invariant(calculate_bitSequence_Rotation_invariant(pad_image,i,j,P,R))
    
    return Rotation_invariant_array

#计算每个像素点领域所构成的比特序列
def calculate_bitSequence_Rotation_invariant(image,i,j,P,R):
    bitSequence=[]
    angle=2*np.pi/P
    center_coordinate=[i+0.5,j+0.5]
    center_value=image[i][j]
    for k in range(0,P):
        if(image[int(center_coordinate[0]-R*np.cos((k+1)*angle))][int(center_coordinate[1]-R*np.sin((k+1)*angle))]>=center_value):
            bitSequence.append(1)
        else:
            bitSequence.append(0)

    return bitSequence

def findMinDecimal_Rotation_invariant(bitSequence):
    values = [] #存放每次移位后的值，最后选择值最小那个
    circle = bitSequence*2  # 用于循环移位，分别计算其对应的十进制
    for i in range(0,8):
        j = 0
        sum = 0
        bit_sum = 0
        while j < 8:
            sum += circle[i+j] << bit_sum
            bit_sum += 1
            j += 1
        values.append(sum)
    return min(values)

##---------------------##
#以下是关于等价LBP的计算
##---------------------##
uniform_map = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 6: 5, 7: 6, 8: 7, 12: 8,14: 9, 15: 10, 16: 11, 24: 12, 28: 13, 30: 14, 31: 15, 32: 16, 48: 17,
                   56: 18, 60: 19, 62: 20, 63: 21, 64: 22, 96: 23, 112: 24,120: 25, 124: 26, 126: 27, 127: 28, 128: 29, 129: 30, 131: 31, 135: 32,143: 33,
                   159: 34, 191: 35, 192: 36, 193: 37, 195: 38, 199: 39, 207: 40,223: 41, 224: 42, 225: 43, 227: 44, 231: 45, 239: 46, 240: 47, 241: 48,
                   243: 49, 247: 50, 248: 51, 249: 52, 251: 53, 252: 54, 253: 55, 254: 56,255: 57}
def LBP_uniform(image):
    original_array=np.zeros(image.shape,dtype=np.uint8)
    H,W=image.shape

    #pad image
    pad_image=np.pad(image,((1,1)),'edge')
    for i in range(1,H+1):
        for j in range(1,W+1):
            bitSequence=calculate_uniform(pad_image,i,j)#获得二进制
            #print(bitSequence)
            decimal=bit2decimal_uniform(bitSequence)
            #print(decimal)
            jumps=calculate_jump(bitSequence)#获得跳变次数
            if(jumps<=2):
                original_array[i-1,j-1]=uniform_map[decimal]
            else:
                original_array[i-1,j-1]=58

    return original_array
#计算每个像素点领域所构成的比特序列
def calculate_uniform(img,i,j):
    sum = []
    if img[i - 1, j ] > img[i, j]:
        sum.append(1)
    else:
        sum.append(0)
    if img[i - 1, j+1 ] > img[i, j]:
        sum.append(1)
    else:
        sum.append(0)
    if img[i , j + 1] > img[i, j]:
        sum.append(1)
    else:
        sum.append(0)
    if img[i + 1, j+1 ] > img[i, j]:
        sum.append(1)
    else:
        sum.append(0)
    if img[i + 1, j ] > img[i, j]:
        sum.append(1)
    else:
        sum.append(0)
    if img[i + 1, j - 1] > img[i, j]:
        sum.append(1)
    else:
        sum.append(0)
    if img[i , j - 1] > img[i, j]:
        sum.append(1)
    else:
        sum.append(0)
    if img[i - 1, j - 1] > img[i, j]:
        sum.append(1)
    else:
        sum.append(0)
    return sum    

#计算跳变次数
def calculate_jump(bitSequence):
    jumps=0
    for i in range(0,len(bitSequence)-1):#有些疑问
        if(bitSequence[i]!=bitSequence[i+1]):
            jumps=jumps+1
    return jumps

#计算比特序列对应的十进制值
def bit2decimal_uniform(bitSequence):
    decimal=0
    #print(bitSequence.shape[1])
    for i in range(0,8):
        decimal=decimal+bitSequence[i]*np.power(2,7-i)
    #print(decimal)
    return decimal

#对于等价LBP，计算直方图
def calculate_histogram_uniform(original_array):
    H,W=original_array.shape
    total=H*W*1
    histogram=[]
    for i in range(0,59):
        histogram.append(np.sum(original_array==i)/total)

    return histogram



#提取LBP特征向量
def vector_extract(image):
    #将检测窗口划分为16×16的小区域（cell）
    H,W=image.shape
    size_H=round(H/16)
    size_W=round(W/16)
    padimage=cv2.resize(image,(16*size_W,16*size_H),interpolation=cv2.INTER_CUBIC)
    #print(padimage.shape)

    #记录所有cell的直方图
    cell_histogram=np.zeros([16,16,59],dtype=np.float64)
    #print(cell_histogram.shape)

    for i in range(0,cell_histogram.shape[0]):
        for j in range(0,cell_histogram.shape[1]):
            #print(padimage[i*size_H:(i+1)*size_H,j*size_W:(j+1)*size_W].shape)
            temp=calculate_histogram_uniform(LBP_uniform(padimage[i*size_H:(i+1)*size_H,j*size_W:(j+1)*size_W]))
            #print(temp)
            for k in range(0,59):
                #print(temp[k])
                cell_histogram[i,j,k]=temp[k]
                #print(cell_histogram[i,j,k])

    LBP_vector=[]
    for i in range(0,cell_histogram.shape[0]):
        for j in range(0,cell_histogram.shape[1]):
            #L2范数归一化
            mag=lambda  vector: math.sqrt( sum(element**2 for element in vector))
            magnitude=mag(cell_histogram[i][j])
            if magnitude!=0:
                nor = lambda vector,magnitude: [element/magnitude for element in vector]
                histogram=nor(cell_histogram[i][j],magnitude)
                LBP_vector.append(histogram)

    return np.asarray(LBP_vector)


def cosine_distance(vector1,vector2):
    molecule=np.sum(vector1*vector2)
    denominator=np.power(np.sum(np.power(vector1,2)),0.5)*np.power(np.sum(np.power(vector2,2)),0.5)
    similarity=molecule/denominator
    distance=1-similarity
    return distance


if  __name__=='__main__':
    #print(uniform_map[2])
    """img=cv2.imread('Q4_2.tif',cv2.IMREAD_GRAYSCALE)
    cv2.imshow('Q4_2',img)
    vector=vector_extract(img)
    print(vector.shape)
    print(vector)"""

    """out=LBP_uniform(img)
    cv2.imshow('LBP_original',out)
    hist1=calculate_histogram_uniform(out)
    LBP_vector=vector_extract(hist1)
    print(LBP_vector.shape)

    out2=LBP_Rotation_invariant(img,8,1)
    cv2.imshow('LBP_r',out2)
    hist2=calculate_histogram_original(out2)

    out_ku=skimage.feature.local_binary_pattern(img,8,1,method='nri_uniform')
    out_ku=out_ku.astype(np.uint8)
    
    cv2.imshow('ku',out_ku)
    out_hist=calculate_histogram_original(out_ku)

    x=np.arange(256)
    y=np.arange(59)
    plt.subplot(3, 1, 1)
    plt.bar(y,hist1)
    plt.subplot(3, 1, 2)
    plt.bar(x,hist2)
    plt.subplot(3, 1, 3)
    plt.bar(x,out_hist)
    plt.show()

    cv2.waitKey(0)"""

    """a=np.zeros([1,8],dtype=np.uint8)
    for i in range(0,a.shape[1]):
        if(i%2==0):
            a[0][i]=1
    print(a)
    b=findMinDecimal_Rotation_invariant(a)
    print(b)
    print(map(35))"""

    template=cv2.imread('BioID_0107.pgm',cv2.IMREAD_GRAYSCALE)
    cv2.imshow('template',template)

    test1=cv2.imread('BioID_0100.pgm',cv2.IMREAD_GRAYSCALE)
    cv2.imshow('test1',test1)

    test2=cv2.imread('BioID_0118.pgm',cv2.IMREAD_GRAYSCALE)
    cv2.imshow('test2',test2)

    time_start_template = time.perf_counter()
    vector_template=vector_extract(template)
    time_end_template = time.perf_counter()
    run_time_template= time_end_template - time_start_template
    print("运行时长：", run_time_template)

    out_template=LBP_uniform(template)
    cv2.imshow('template',out_template)
    hist_template=calculate_histogram_uniform(out_template)


    time_start_test1 = time.perf_counter()
    vector_test1=vector_extract(test1)
    time_end_test1 = time.perf_counter()
    run_time_test1= time_end_test1 - time_start_test1
    print("运行时长：", run_time_test1)

    out_test1=LBP_uniform(test1)
    cv2.imshow('test1',out_test1)
    hist_test1=calculate_histogram_uniform(out_test1)

    time_start_test2 = time.perf_counter()
    vector_test2=vector_extract(test2)
    time_end_test2 = time.perf_counter()
    run_time_test2= time_end_test2 - time_start_test2
    print("运行时长：", run_time_test2)

    out_test2=LBP_uniform(test2)
    cv2.imshow('test2',out_test2)
    hist_test2=calculate_histogram_uniform(out_test2)

    y=np.arange(59)
    plt.subplot(3, 1, 1)
    plt.bar(y,hist_template)
    plt.subplot(3, 1, 2)
    plt.bar(y,hist_test1)
    plt.subplot(3, 1, 3)
    plt.bar(y,hist_test2)
    plt.show()

    distance1=cosine_distance(vector_template,vector_test1)
    distance2=cosine_distance(vector_template,vector_test2)
    distance3=cosine_distance(vector_test2,vector_test1)

    print(distance1)
    print(distance2)
    print(distance3)

    cv2.waitKey(0)
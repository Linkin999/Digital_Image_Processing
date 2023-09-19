import time
import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
import face_recognition
from skimage import feature as ft

class Hog_descriptor():
    #------#
    #initialize
    # cell_size cell的高度和宽度
    #bin_size 角度区间份数
    #------#
    def __init__(self,img,cell_size,bin_size):
        self.img=cv2.resize(img,(64,128),interpolation=cv2.INTER_CUBIC)#把图像缩放成128×64

        ##gamma normalize
        self.img=np.sqrt(img*1.0/float(np.max(img)))
        self.img=self.img*255

        self.cell_size=cell_size
        self.bin_size=bin_size
        self.angle_unit=180/self.bin_size

    #---------#
    # get hog vector 
    #---------#
    def extract(self):
        height,width=self.img.shape

        ##计算梯度大小和角度
        gradient_magnitude, gradient_direction=self.global_gradient()
        gradient_magnitude=abs(gradient_magnitude)

        #cell_gradient_vector用来保存每个cell的梯度向量
        cell_gradient_vector=np.zeros([int(height/self.cell_size),int(width/self.cell_size),self.bin_size],dtype=np.float64)
        height_cell_vector,width_cell_vector,_=cell_gradient_vector.shape

        #计算每个细胞的梯度直方图
        for i in range(0,height_cell_vector):
            for j in range(0,width_cell_vector):
                #获得该细胞的梯度大小
                cell_magnitude=gradient_magnitude[i*self.cell_size:(i+1)*self.cell_size,j*self.cell_size:(j+1)*self.cell_size]
                #获得该细胞的梯度角度
                cell_direction=gradient_direction[i*self.cell_size:(i+1)*self.cell_size,j*self.cell_size:(j+1)*self.cell_size]

                #转换为梯度直方图格式
                cell_gradient_vector[i][j]=self.cell_gradient(cell_magnitude,cell_direction)

        #hog图像
        hog_image=self.render_gradient(np.zeros([height,width]),cell_gradient_vector)

        hog_vector=[]

        #block为2×2,减1的目的是block只有（height_cell_vector-1）×（width_cell_vector-1）
        for i in range(0,height_cell_vector-1):
            for j in range(0,width_cell_vector-1):
                block_vector=[]
                #extend并不改变维度大小
                block_vector.extend(cell_gradient_vector[i,j])
                block_vector.extend(cell_gradient_vector[i,j+1])
                block_vector.extend(cell_gradient_vector[i+1,j])
                block_vector.extend(cell_gradient_vector[i+1,j+1])

                #L2范数归一化
                mag=lambda  vector: math.sqrt( sum(element**2 for element in vector))#这里有点不太一样
                magnitude=mag(block_vector)+1e-5
                if magnitude!=0:
                    nor = lambda vector,magnitude: [element/magnitude for element in vector]
                    block_vector=nor(block_vector,magnitude)
                hog_vector.append(block_vector)

        return np.asarray(hog_vector),hog_image
    

    #-----#
    #计算每一个像素点的梯度大小和角度
    #-----#
    def global_gradient(self):
        #用sobel算子进行梯度计算，CV_64F表示数据类型为float64,(1,0)代表求x方向的一阶导数
        #（0，1）代表求y方向上的一阶导数，ksize代表Sobel算子的大小
        gradient_value_x=cv2.Sobel(self.img,cv2.CV_64F,1,0,ksize=3)
        gradient_value_y=cv2.Sobel(self.img,cv2.CV_64F,0,1,ksize=3)

        gradient_magnitude=np.sqrt(np.power(gradient_value_x,2)+np.power(gradient_value_y,2))
        gradient_direction=cv2.phase(gradient_value_x,gradient_value_y,angleInDegrees=True)
        for i in range(0,gradient_direction.shape[0]):
            for j in range(0,gradient_direction.shape[1]):
                if(gradient_direction[i][j]>180.0):
                    gradient_direction[i][j]=gradient_direction[i][j]-180.0
        #gradient_direction=np.arctan(gradient_value_y/gradient_value_x)
        return gradient_magnitude, gradient_direction

    #将梯度大小分解到方向上
    def cell_gradient(self,cell_magnitude,cell_direction):
        orientation=[0]*self.bin_size
        for i in range(0,cell_magnitude.shape[0]):
            for j in range(0,cell_magnitude.shape[1]):
                gradient_strength=cell_magnitude[i][j]
                gradient_angle=cell_direction[i][j]
                left_edge_angle,right_edge_angle,mod=self.get_interval_bins(gradient_angle)
                orientation[left_edge_angle]=orientation[left_edge_angle]+gradient_strength*(1-(mod/self.angle_unit))
                orientation[right_edge_angle]=orientation[right_edge_angle]+gradient_strength*(mod/self.angle_unit)
        return orientation
    
    #计算像素点所属的角度
    def get_interval_bins(self,gradient_angle):
        index=int(gradient_angle/self.angle_unit)
        mod=gradient_angle%self.angle_unit
        return index%self.bin_size,(index+1)%self.bin_size,mod
        #(index+1)%self.bin_size是为了防止超出边界

    #绘制梯度直方图
    def render_gradient(self,image,cell_gradient_vector):
        cell_width=int(self.cell_size/2)#这里不太一样
        max_magnitude=np.array(cell_gradient_vector).max()

        for x in range(cell_gradient_vector.shape[0]):
            for y in range(cell_gradient_vector.shape[1]):
                cell_gradient=cell_gradient_vector[x][y]
                #归一化
                cell_gradient=cell_gradient/max_magnitude
                angle=0

                for magnitude in cell_gradient:
                    angle_radian=math.radians(angle)
                    x1=int(x*self.cell_size+magnitude*cell_width*math.cos(angle_radian))
                    y1=int(y*self.cell_size+magnitude*cell_width*math.sin(angle_radian))
                    x2=int(x*self.cell_size-magnitude*cell_width*math.cos(angle_radian))
                    y2=int(y*self.cell_size-magnitude*cell_width*math.sin(angle_radian))
                    cv2.line(image,(y1,x1),(y2,x2),int(255*math.sqrt(magnitude)))

        return image

def Euclidean_distance(vector1,vector2):
    difference=vector1-vector2
    square=np.power(difference,2)
    distance=np.power(np.sum(square),0.5)
    return distance

def cosine_distance(vector1,vector2):
    molecule=np.sum(vector1*vector2)
    denominator=np.power(np.sum(np.power(vector1,2)),0.5)*np.power(np.sum(np.power(vector2,2)),0.5)
    similarity=molecule/denominator
    distance=1-similarity
    return distance


if  __name__=='__main__':
    template=cv2.imread('BioID_0107.pgm',cv2.IMREAD_GRAYSCALE)
    cv2.imshow('template',template)

    test1=cv2.imread('BioID_0100.pgm',cv2.IMREAD_GRAYSCALE)
    cv2.imshow('test1',test1)

    test2=cv2.imread('BioID_0118.pgm',cv2.IMREAD_GRAYSCALE)
    cv2.imshow('test2',test2)

    #提取人脸
    template_location=face_recognition.face_locations(template)
    template_face=template[template_location[0][0]:template_location[0][2],template_location[0][3]:template_location[0][1]]
    #cv2.imshow('template_face',template_face)

    test1_location=face_recognition.face_locations(test1)
    test1_face=test1[test1_location[0][0]:test1_location[0][2],test1_location[0][3]:test1_location[0][1]]
    #cv2.imshow('test1_face',test1_face)

    test2_location=face_recognition.face_locations(test2)
    test2_face=test2[test2_location[0][0]:test2_location[0][2],test2_location[0][3]:test2_location[0][1]]
    #cv2.imshow('test2_face',test2_face)

    #HOG
    time_start_template = time.perf_counter()
    hog_template=Hog_descriptor(template,cell_size=8,bin_size=9)
    hog_vector_template,hog_image_template=hog_template.extract()
    time_end_template = time.perf_counter()
    run_time_template= time_end_template - time_start_template
    print("运行时长：", run_time_template)

    
    cv2.imshow('hog_template',hog_image_template)

    out2_template=ft.hog(template,orientations=9,pixels_per_cell=[8,8],cells_per_block=[2,2],visualize=True)
    cv2.imshow("hog_template",out2_template[1])
    

    time_start_test1 = time.perf_counter()
    hog_test1=Hog_descriptor(test1,cell_size=8,bin_size=9)
    hog_vector_test1,hog_image_test1=hog_test1.extract()
    time_end_test1 = time.perf_counter()
    run_time_test1= time_end_test1 - time_start_test1
    print("运行时长：", run_time_test1)

    cv2.imshow('hog_test1',hog_image_test1)
    out2_test1=ft.hog(test1,orientations=9,pixels_per_cell=[8,8],cells_per_block=[2,2],visualize=True)
    cv2.imshow("hog_test1",out2_test1[1])

    time_start_test2 = time.perf_counter()
    hog_test2=Hog_descriptor(test2,cell_size=8,bin_size=9)
    hog_vector_test2,hog_image_test2=hog_test2.extract()
    time_end_test2 = time.perf_counter()
    run_time_test2= time_end_test2 - time_start_test2
    print("运行时长：", run_time_test2)

    cv2.imshow('hog_test2',hog_image_test2)

    out2_test2=ft.hog(test2,orientations=9,pixels_per_cell=[8,8],cells_per_block=[2,2],visualize=True,feature_vector=True)
    cv2.imshow("hog_test2",out2_test2[1])

    cv2.waitKey(0)

    #距离
    distance1=cosine_distance(hog_vector_template,hog_vector_test1)
    distance2=cosine_distance(hog_vector_template,hog_vector_test2)
    distance3=cosine_distance(hog_vector_test1,hog_vector_test2)
    print(distance1)
    print(distance2)
    print(distance3)

    distance11=cosine_distance(out2_template[0],out2_test1[0])
    distance22=cosine_distance(out2_template[0],out2_test2[0])
    distance33=cosine_distance(out2_test2[0],out2_test1[0])
    """print(distance11)
    print(distance22)
    print(distance33)"""


    """img=cv2.imread('Q4_2.tif',cv2.IMREAD_GRAYSCALE)
    #print(img.shape)
    cv2.imshow('Q4_2',img)

    #提取人脸
    img2=face_recognition.face_locations(img)
    #print(img2)
    #print(img2[0][0])
    #print(img2[0][1])
    #print(img2[0][2])
    #print(img2[0][3])
    out=img[img2[0][0]:img2[0][2],img2[0][3]:img2[0][1]]
    out3=Sobel_spatial(img)
    #print(out)
    cv2.imshow('face',out)
    cv2.imshow('face2',out3)

    hog=Hog_descriptor(out,cell_size=8,bin_size=9)
    hog_vector,hog_image=hog.extract()
    plt.imshow(hog_image,cmap=plt.cm.gray)
    plt.show()
    print(hog_vector)
    print(hog_vector.shape)

    hog3=Hog_descriptor(out3,cell_size=8,bin_size=9)
    hog_vector3,hog_image3=hog3.extract()
    plt.imshow(hog_image3,cmap=plt.cm.gray)
    plt.show()
    print(hog_vector3)
    print(hog_vector3.shape)


    out2=ft.hog(out,orientations=9,pixels_per_cell=[8,8],cells_per_block=[2,2],visualize=True)
    plt.imshow(out2[1],cmap=plt.cm.gray)
    plt.show()
    print(out2[0])
    print(out2[0].shape)

    out4=ft.hog(out3,orientations=9,pixels_per_cell=[8,8],cells_per_block=[2,2],visualize=True)
    plt.imshow(out4[1],cmap=plt.cm.gray)
    plt.show()"""

    #cv2.imshow(img[img2[0][0]:img2[0][2]+1][img2[0][3]:img2[0][1]+1])

    """hog=Hog_descriptor(img,cell_size=8,bin_size=9)
    hog_vector,hog_image=hog.extract()
    plt.imshow(hog_image,cmap=plt.cm.gray)
    plt.show()

    print('hog_vector',hog_vector.shape)
    print('hog_image',hog_image.shape)"""
    
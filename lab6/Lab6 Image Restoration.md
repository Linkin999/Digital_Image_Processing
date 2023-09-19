<h1 align = "center">Lab6 Image Restoration</h1>

<center>张旭东 12011923</center>

## 1.Introduction

​	As an important part of image processing, image restoration technology is a key issue explored by many scholars at home and abroad. Image restoration technology involves three aspects: the establishment of image restoration model, the use of image restoration algorithm and the setting of image quality measurement index. By changing the imaging model, degradation space domain, optimization criteria and methods of degraded images, different image restoration methods will be formed. The purpose of image restoration is to remove or reduce the degradation that occurs during the acquisition of digital images, and to make a best estimate of the original image, which can be used for deblurring, improving resolution and contrast, aerial reconnaissance, remote sensing, radiation imaging.

​	The filters used in image restoration contains spatial domain filters and frequency domain filters. Spatial domain filters used in this lab include arithmetic mean filter, contraharmonic mean filter, median filter, max filter, min filter, midpoint filter, alpha-trimmed mean filter and adaptive mean filter. Frequency domain filters used in this lab include full inverse filter, radially limited inverse filter, Wiener filter and constraint least square filter.

​	Among the mentioned spatial domain filters, removing different types of noise require different kinds of spatial domain filters. Contraharmonic mean filter is well suited for reducing the effects of salt-and-pepper noise. When pepper noise is present, contraharmonic mean filter and max filter are more effective. When When salt noise is present, contraharmonic mean filter and min filter are more effective. Median filter is more effective in the presence of both bipolar and unipolar impulse noise, like pepper-and-salt noise. Midpoint filter is best for randomly distributed noise, like Gaussian or uniform noise. Alpha-trimmed mean filter is useful in situations involving multiple types of noise, such as combination of salt-and-pepper and Gaussian noise. What's more, the behavior of adaptive mean filter changes based on statistical characteristics of the image inside the filter region defined by the $m×n$ rectangular window, so the performance is superior to that of the filters dicussed.

​	Among the mentioned frequency domain filters,full inverse filter can't exactly recover the undegraded image because the spectrum of noise is unknown and have probability to make noise dominate the estimation. One approach is to limit the filter frequencies to values near the origin by adding a lowpass filter. This kind of approach is called radially limited inverse filter. The principle of Wiener filter is to find an estimate of the uncorrupted image such that the mean square error between them is minimized. To get better result, parameter in Wiener need to be manually adjusted. To solve the problem where the power spectra of the undegraded image and noise must be known in Wiener filter, constrained least squares filter has been proposed. It just requires the mean and variance of the noise. However, in this lab, the overall effect of constraint least square filter is worse than that of Wiener filter.   

## 2.Analysis and Result

### 2.1 Restoration in the Presence of Noise Only

​	**principle:** The model of image restoration in the presence of Noise Only is
$$
g(x,y)=f(x,y)+\eta(x,y)\\
G(\mu,\nu)=F(\mu,\nu)+N(\mu,\nu)
$$
​	**Question Formulation:** There are several spatial filters used in image restoration in the presence of noise only, including arithmetic mean filter, Contraharmonic mean filter and adaptive mean filter. Let $S_{xy}$ represent the set of coordinates in a rectangle subimage window of size $m×n$, centered at $(x,y)$.

​	Arithmetic mean filter:
$$
\widehat f(x,y)=\frac{1}{mn}\sum_{(s,t)\in S_{xy}}g(s,t)
$$
​	Geometric mean filter:
$$
\widehat f(x,y)=[\prod_{(s,t)\in S_{xy}}g(s,t)]^{\frac{1}{mn}}
$$
​	Generally, a geometric mean filter achieves smoothing comparable to the arithmetic mean filter, but it tends to lose less image detail in the process.

​	Contraharmonic mean filter:
$$
\widehat f(x,y)=\frac{\sum_{(s,t)\in S_{xy}}g(s,t)^{Q+1}}{\sum_{(s,t)\in S_{xy}}g(s,t)^{Q}}
$$
​	$Q$ is the order of the filter. It is well suited for reducing the effects of salt-and-pepper noise. $Q>0$ for pepper noise and $Q<0$ for salt noise.

​	Median filter:
$$
\widehat f(x,y)=median_{(s,t)\in S_{xy}}[g(s,t)]
$$
​	It is effective in the presence of both bipolar and unipolar impulse noise.

​	Max filter:
$$
\widehat f(x,y)=max_{(s,t)\in S_{xy}}[g(s,t)]
$$
​	It is effective in the presence of pepper noise.	

​	Min filter:
$$
\widehat f(x,y)=min_{(s,t)\in S_{xy}}[g(s,t)]
$$
​	It is effective in the presence of salt noise.

​	Midpoint filter:
$$
\widehat f(x,y)=\frac{1}{2}[max_{(s,t)\in S_{xy}}[g(s,t)]+min_{(s,t)\in S_{xy}}[g(s,t)]]
$$
​	It is best for randomly distributed noise, like Gaussian or uniform noise.	

​	Alpha-trimmed mean filter:
$$
\widehat f(x,y)=\frac{1}{mn-d}\sum_{(s,t)\in S_{xy}}g_{r}(s,t)
$$
​	The $\frac{d}{2}$ lowest and the $\frac{d}{2}$ highest intensity values of $g(s,t)$ in the neighborhood $S_{xy}$. Let $g_{r}(s,t)$ represent the remaining $mn-d$ pixels. I t is useful in simulations involving multiple types of noise, such as a combination of salt-and -pepper and Gaussian noise.

​	Adaptive filter: The behavior changes based on statistical characteristics of the image inside the filter region defined by the $m×n$ rectangular window. The performance is superior to that of the filters discussed. Let $S_{xy}$ denotes local region and the response of the filter at the center point $(x,y)$ of $S_{xy}$ is based on four quantities:

- $g(x,y)$, the value of the noisy image at $(x,y)$;
- $\sigma^{2}_{\eta}$, the variance of the noise corrupting $f(x,y)$ to form $g(x,y)$;
- $m_{L}$, the local mean of the pixels in $S_{xy}$;
- $\sigma_{L}^{2}$, the local variance of the pixels in $S_{xy}$;

​	The behavior filter of the filter is:

- if $\sigma^{2}_{\eta}$ is zero, the filter should return simply the value of $g(x,y)$.
- if the local variance is high relative to $\sigma^{2}_{\eta}$, the filter should return a value close to $g(x,y)$;
- if the two variances are equal, the filter returns the arithmetic mean value of the pixels in $S_{xy}$.

​	An adaptive expression for obtaining $\widehat f(x,y)$ based on these assumption may be given by
$$
\widehat f(x,y)=g(x,y)-\frac{\sigma^{2}_{\eta}}{\sigma^{2}_{L}}[g(x,y)-m_{L}]
$$
​	An tacit assumption in the above equation is that $\sigma_{\eta}^{2} \leq \sigma_{L}^{2}$.

​	For adaptive median filter, $z_{min}=$ minimum intensity value in $S_{xy}$, $z_{max}=$ maximum intensity value in $S_{xy}$,  $z_{med}=$ median intensity value in $S_{xy}$,  $z_{xy}=$ intensity value at coordinate $(x,y)$ and $S_{max}=$ maximum allowed size of $S_{xy}$. The adaptive median-filtering works in two stages:

​	stage A:

​			$A1=z_{med}-z_{min}$; $A2=z_{med}-z_{max}$

​			if $A1>0$ and $A2<0$, go to stage B

​			else increase the window size

​			if window size $\leq S_{max}$, repeat stage A; else output $z_{med}$

​	stage B:

​			$B1=z_{xy}-z_{min}$; $B2=z_{xy}-z_{max}$ 

​			if $B1>0$ and $B2<0$, output $z_{xy}$; else output $z_{med}$. 

**Experiment1:**

​	From the `Q6_1_1.tif`, `Q6_1_2.tif` and lecture notes, it is obvious that `Q6_1_1.tif` is corrupted by pepper noise with a probability of $0.1$ and `Q6_1_2.tif` is corrupted by salt noise with a probability of $0.1$. Contraharmonic filter is well suited for reducing the effects of salt-and-pepper noise. $Q>0$ for pepper noise and $Q<0$ for salt noise. Max filter is effective in the presence of pepper noise and min filter is effective in the presence of salt noise. So contraharmonic filter, max filter and min filter are used.

<center class="half">    
    <img src="./Lab6 Image Restoration/image-20230503043000003.png" alt="image-20230503043000003" style="zoom:100%;"width="300"/>    
    <img src="./Lab6 Image Restoration/image-20230503043024121.png" alt="image-20230503043024121" style="zoom:100%;" width="300"/> 
</center

<div align = 'center'><b>Fig.1 Q6_1_1.tif(left) and Q6_1_2.tif(right)</div>

​	**Python code:**

```python
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
```

​	For `Q6_1_1.tif`, max filter with size $3×3$ and contraharmonic filter with order $1.5$ are used. The result is in `Fig2`.

<center class="half">    
    <img src="./Lab6 Image Restoration/image-20230503045901617.png" alt="image-20230503045901617" style="zoom:100%;"width="300"/>    
    <img src="./Lab6 Image Restoration/image-20230503050359056.png" alt="image-20230503050359056" style="zoom:100%;" width="300"/> 
</center

<div align = 'center'><b>Fig.2 Q6_1_1 processed by contraharmonic mean filter(left) and Q6_1_1 processed max filter(right)</div>

​	From the above figure, the size of white spot marked processed by contraharmonic mean filter with order $1.5$ is smaller than that processed by max filter with size $3×3$. The black lines marked  processed by contraharmonic mean filter with order $1.5$  diverge more than that processed by max filter with size $3×3$.

​	For `Q6_1_2.tif`, min filter with size $3×3$ and contraharmonic filter with order $-1.5$ are used. The result is in `Fig3`.

<center class="half">    
    <img src="./Lab6 Image Restoration/image-20230503053147767.png" alt="image-20230503053147767" style="zoom:100%;"width="300"/>    
    <img src="./Lab6 Image Restoration/image-20230503053223328.png" alt="image-20230503053223328" style="zoom:100%;" width="300"/> 
</center

<div align = 'center'><b>Fig.3 Q6_1_2 processed by contraharmonic mean filter(left) and Q6_1_2 processed min filter(right)</div>

​	From the above figure, the size of white spot marked processed by contraharmonic mean filter with order $-1.5$ is smaller than that processed by min filter with size $3×3$. What's more, the black lines processed by contraharmonic mean filter with order $-1.5$ and min filter with size $3×3$ are thicker and clearer than that processed by contraharmonic mean filter with order $1.5$ and max filter with size $3×3$.

**algorithm complexity:** 

|                | contraharmonic(order=1.5,3*3) | max(3*3)   | min(3*3)   |
| :------------: | ----------------------------- | ---------- | ---------- |
| average time/s | 12.4599585                    | 1.41806620 | 1.39890000 |
|   space/byte   | f~11MN                        | f~9MN      | f~9MN      |

​	It is obvious that the average time used by contraharmonic mean filter is longer than that used by max filter and min filter. The main time in contraharmonic mean filter is used in exponential operation while that in max filter or min filter is used in sort operation. The occupied space in contraharmonic mean filter is proportional to $11MN$, which is used to store `padimage` and the values of `operator1` and `operator2` every loop.($M$, $N$ is the size of original image.) The occupied space in max and min filter is proportional to $9MN$, which is used to store `padimage`. 

​	**Experiment2:**

​	For  `Q6_1_3.tif` and lecture notes, it is obvious that the image is corrupted by salt-and-pepper noise with a high probabilities. Median filter is effective in the presence of both bipolar and unipolar impulse noise. Besides, the behavior of adaptive mean filter changes based on statistical characteristics of the image inside the filter region defined by the $m×n$ rectangular window and the performance is superior to that of the filters discussed. So median filter and adaptive median filter are used.

<img src="./Lab6 Image Restoration/image-20230503074526774.png" alt="image-20230503074526774" style="zoom: 67%;" />

<div align = 'center'><b>Fig.4 Q6_1_3 </div>

​	**Pseudo code for adaptive median filter:** 

```python
padimage=np.pad(input_image,((H,H),(W,W)),'symmetric')
for i in range(H,2*H):
        for j in range(W,2*W):
			local_image=padimage[i-int((S-1)/2):i+int((S-1)/2)+1,j-int((S-1)/2):j+int((S-1)/2)+1]
            output_image[i-H,j-W]=stage_A
Stage A: A1=Zmed-Zmin, A2=Zmed-Zmax
    	if(A1>0 and A2<0)
			go to stage B
         else
            if(window size<Smax)
            	stage A
            else
            	output Zmed
Stage B:B1=Zxy-Zmin, B2=Zxy-Zmax
		if(	B1>0 and B2<0)
    		output Zxy
        else
        	output Zmed
```

​	**Python code:**

```python
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
```

​	For  `Q6_1_3.tif`, median filter with $7×7$ and adaptive median filter with $S_{max}=7$ are used. The result is shown in `Fig 5`.

<center class="half">    
    <img src="./Lab6 Image Restoration/image-20230503080803617.png" alt="image-20230503080803617" style="zoom:100%;"width="300"/>    
    <img src="./Lab6 Image Restoration/image-20230503080850539.png" alt="image-20230503080850539" style="zoom:100%;" width="300"/> 
</center

<div align = 'center'><b>Fig.5 Q6_1_3 processed by median filter(left) and Q6_1_3 processed by adaptive median filter(right)</div>

​	It is obvious that the result processed by adaptive filter is much clearer than that processed by median filter. The result processed by adaptive filter preserve more edge details than that processed by median filter, which proves the performance of adaptive is superior to that of the filters discussed.

​	**algorithm complexity:** 

|                | adaptive median(Smax=7)    | median(7×7)  |
| -------------- | -------------------------- | :----------: |
| average time/s | 8.610005                   |  4.4872866   |
| space          | (3^ 2+10)MN<=f<=(7^2+10)MN | f~(7^2+10)MN |

​	It is obvious that the average time adaptive median filter with $S_{max}=7$ take is longer than that median filter with size $7×7$ take. Although the size of local image generated by adaptive median filter with $S_{max}=7$ is smaller than that generated by median filter with size $7×7$ overall, the time used by `if` statement make the average time adaptive median filter take longer. That is one of reasons why the average time adaptive median filter with $S_{max}=7$ take is longer than that median filter with size $7×7$ take. As for space complexity, the space complexity of adaptive median filter is less than that of median filter. The number of local images generated by both of them are the same. However, the size of local image generated by median filter is $7×7$ while that generated by adaptive median filter is less than $7×7$ on average. $10 MN$ is the total size of `padimage` and `outputimage`.

​	**Experiment3:**

​	For `Q6_1_4.tif` and lecture notes, it is easy to know that the image is corrupted by salt-and-pepper noise. According to the `experiment2`, median filter and adaptive median filter are used.

<img src="./Lab6 Image Restoration/image-20230503162935296.png" alt="image-20230503162935296" style="zoom: 67%;" />

<div align = 'center'><b>Fig.6 Q6_1_4 </div>

​	The result is in `Fig 7`.

<center class="half">    
    <img src="./Lab6 Image Restoration/image-20230503164934019.png" alt="image-20230503164934019" style="zoom:100%;"width="300"/>    
    <img src="./Lab6 Image Restoration/image-20230503165020400.png" alt="image-20230503165020400" style="zoom:100%;" width="300"/> 
</center

<center class="half">    
    <img src="./Lab6 Image Restoration/image-20230503165047648.png" alt="image-20230503165047648" style="zoom:100%;"width="300"/>    
    <img src="./Lab6 Image Restoration/image-20230503164439779.png" alt="image-20230503164439779" style="zoom:100%;" width="300"/> 
</center

<div align = 'center'><b>Fig.7 result processed by median filter with 3×3(Upper left), median filter with 5×5(Upper right),median filter with 7×7(lower left) and adaptive median filter with Smax=7(lower right) </div>

​	From the above figures, what can be found is that the performance of median filter with size $5×5$ is the best among the three median filters of different sizes. Besides, the performance of adaptive median filter with $S_{max}=7$ is the best among the four filters because it retains boundary information when removing salt-and-pepper noise. However, all of the above four results still contain other noise. According to the distribution of noise in the above results, it can be judged that it is additive uniform noise. According to the introduction of midpoint filter, which is best for randomly distributed noise,like Gaussian or uniform noise, midpoint filter is used to process the results processed by median filter with $5×5$ and adaptive median filter with $S_{max}=7$. Also, according to the introduction of alpha-trimmed mean filter, which is useful in situations involving multiple types of noise, such as a combination of salt-and-pepper and Gaussian noise, alpha-trimmed mean filter with size $5×5$ and $d=10$ is used to filter the `Q6_1_4.tif`. 

​	**Python code:**

```python
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

```

​	The results are in `Fig 8`.

<center class="half">    
    <img src="./Lab6 Image Restoration/image-20230503173220576.png" alt="image-20230503173220576" style="zoom:100%;"width="300"/>    
    <img src="./Lab6 Image Restoration/image-20230503173259650.png" alt="image-20230503173259650" style="zoom:100%;" width="300"/> 
</center

<img src="./Lab6 Image Restoration/image-20230503173405240.png" alt="image-20230503173405240" style="zoom: 55%;" />

<div align = 'center'><b>Fig.8 result processed by alpha-trimmed mean filter with size 5×5 and d=10(Upper left), midpoint filter with 3×3 followed by median filter with 5×5(Upper right),midpoint filter with 3×3 followed by adaptive median filter with Smax=7(lower) </div>

​	According to the above three pictures, the performance of midpoint filter with $3×3$ followed by adaptive median filter with $S_{max}=7$ is the best among them. The result processed by alpha-trimmed mean filter with size $5×5$ and $d=10$ still contains some pepper noise. Compared to the results processed by median filter with size $5×5$ and adaptive median filter with $S_{max}=7$, the results processed by midpoint filter with $3×3$ followed by median filter with $5×5$ and midpoint filter with $3×3$ followed by adaptive median filter with $S_{max}=7$ contain less additive uniform noise. However, the results loss some edge information at the same time, compared to the results processed by median filter with size $5×5$ and adaptive median filter with $S_{max}=7$. Compared to the result processed by midpoint filter with $3×3$ followed by median filter with $5×5$, that processed by midpoint filter with $3×3$ followed by adaptive median filter with $S_{max}=7$ is more clear.

​	**algorithm complexity:** 

|                | alpha-trimmed mean(5*5,10) | midpoint with median(3×3,5×5) | midpoint with adaptive(3*3,7)           |
| -------------- | :------------------------: | ----------------------------- | --------------------------------------- |
| average time/s |         3.1907555          | 7.6776330999999995            | 10.369266600000001                      |
| space          |        f~(5×5+10)MN        | f~(5×5+10+3×3+10)MN           | (5×5+10+3×3+10)MN<=f<=(7×7+10+3×3+10)MN |

​	It is obvious that the average time used by midpoint filter followed by adaptive median filter is the longest, followed by midpoint filter with median filter, and the last is alpha-trimmed mean filter. The reason why the average time used by  midpoint filter followed by adaptive median filter and midpoint filter followed by median filter is much longer than that used by alpha-trimmed mean filter is that the first two algorithms contains $8$ `for` loops while the latter algorithm contains $4$ `for` loop. For alpha-trimmed mean filter, $5×5MN$ is the total space occupied by local image and $10MN$ is the total size of `padimage` and `outputimage`. For midpoint filter with median filter and midpoint filter with adaptive median filter, $(3×3+10)MN$ is the total space occupied in midpoint filter algorithm. Among the three algorithm, the space complexity of midpoint filter with adaptive median filter is the highest.

### 2.2 Image restoration   

​	A model of the image degradation/restoration process is

<img src="./Lab6 Image Restoration/image-20230503191926712.png" alt="image-20230503191926712" style="zoom:67%;" />

<div align = 'center'><b>Fig.9 A model of the image degradation/restoration process </div>

​	
$$
g(x,y)=H[f(x,y)]+\eta (x,y)
$$
​	If $H$ is a linear, position-invariant process, then the degraded image is given in the spatial domain by
$$
g(x,y)=h(x,y) \bigotimes f(x,y)+\eta(x,y)
$$
​	The degraded image in frequency domain is
$$
G(\mu,\nu)=H(\mu,\nu)F(\mu,\nu)+N(\mu,\nu)
$$
​	There are three principal ways to estimate the degradation function, including observation, experimentation and mathematical modeling.

​	**Observation:** take a small area where the signal is strong.
$$
H_{s}(\mu,\nu)=\frac{G_{s}(\mu,\nu)}{\widehat F_{s}(\mu,\nu)}
$$
​	**Experimentation:** similar equipment acquiring the degraded image is available.
$$
H(\mu,\nu)=\frac{G(\mu,\nu)}{A}
$$
​	**Mathematical modeling:**

​		A model about atmospheric turbulence
$$
H(\mu,\nu)=e^{-k(\mu ^{2}+\nu ^{2})^{\frac{5}{6}}}
$$
​		where $k$ is a constant that depends on the nature of the turbulence.

​		An image blurred by uniform linear motion between the image and the sensor during image acquisition.
$$
H(\mu,\nu)=\frac{T}{\pi (\mu a+\nu b)}\sin[\pi(\mu a+\nu b)]e^{-j\pi(\mu a+\nu b)}
$$
​	In this task, full inverse filter, radially limited inverse filter and Wiener filter will be introduced. They are all frequency domain filters.

​	**Full inverse filter:** An estimate of the transform of the original image
$$
\begin{align*}
\widehat F(\mu,\nu)&=\frac{G(\mu,\nu)}{H(\mu,\nu)}\\
&=\frac{F(\mu,\nu)H(\mu,\nu)+N(\mu,\nu)}{H(\mu,\nu)}\\
&=F(\mu,\nu)+\frac{N(\mu,\nu)}{H(\mu,\nu)}
\end{align*}
$$
​	The undegraded image can't be exactly recovered because $N(\mu,\nu)$ is not known. Also, if the degradation function has zero or very small values, then the ratio $\frac{N(\mu,\nu)}{H(\mu,\nu)}$ could easily dominate the estimate $\widehat F(\mu,\nu)$.

​	**Radially limited inverse filter:** Based on full inverse filter, radially limited inverse filter puts a limit on the size of the frequency, which is equivalent to a series of full inverse filter and a low-pass filter.

​	**Wiener filter:** Wiener filter, which is also called minimum mean square error filter, is to find an estimate of the uncorrupted image such that the mean square error between them is minimized. The minimum of the error function is given in the frequency domain by the expression
$$
\begin{align*}
\widehat F(\mu,\nu)&=[\frac{H^{*}(\mu,\nu)S_{f}(\mu,\nu)}{S_{f}(\mu,\nu)|H(\mu,\nu)|^{2}+S_{\eta}(\mu,\nu)}]G(\mu,\nu)\\
&=[\frac{H^{*}(\mu,\nu)}{|H(\mu,\nu)|^{2}+S_{\eta}(\mu,\nu)/S_{f}(\mu,\nu)}]G(\mu,\nu)\\
&=[\frac{1}{H(\mu,\nu)}\frac{|H(\mu,\nu)|^{2}}{|H(\mu,\nu)|^{2}+S_{\eta}(\mu,\nu)/S_{f}(\mu,\nu)}]G(\mu,\nu)
\end{align*}
$$
​	When the power spectrum of $|N(\mu,\nu)|^{2}$ and $|F(\mu,\nu)|^{2}$ are unknown, the following approximation usually is used
$$
\widehat F(\mu,\nu)=[\frac{1}{H(\mu,\nu)}\frac{|H(\mu,\nu)|^{2}}{|H(\mu,\nu)|^{2}+K}]G(\mu,\nu)
$$
​	Image `Q6_2.tif` was degraded from an original image due to the atmosphere turbulence with $k=0.0025$. According to these information, the $k$ in `Formula (16)` is $0.0025$.

<img src="./Lab6 Image Restoration/image-20230503200622390.png" alt="image-20230503200622390" style="zoom:67%;" />

<div align = 'center'><b>Fig.10 Q6_2.tif </div>

​	**Python code:**

```python
def full_inverse_filter(input_image,k):
    input_image_DFT=np.fft.fft2(input_image)
    input_image_center=np.fft.fftshift(input_image_DFT)
    row,col=input_image_center.shape
    H=np.zeros([row,col],dtype=np.complex128)
    for i in range (0,row):
        for j in range(0,col):
            d=np.power(i-row/2,2)+np.power(j-col/2,2)
            H[i,j]=np.exp(-(k*(np.power(d,5/6))))
    output_image=input_image_center/H
    output_image=np.fft.ifftshift(np.fft.ifft2(output_image))
    output_image=np.real(output_image)

    a=np.max(output_image)
    b=np.min(output_image)
    for i in range(0,output_image.shape[0]):
        for j in range(0,output_image.shape[1]):
            output_image[i,j]=int((output_image[i,j]-b)*255/(a-b))
    
    output_image=np.array(output_image,dtype=np.uint8)
    return output_image

def limited_inverse_filter(input_image,radius,k,order):
    input_image_DFT=np.fft.fft2(input_image)
    input_image_center=np.fft.fftshift(input_image_DFT)
    row,col=input_image_center.shape

    H=np.zeros([row,col],dtype=np.complex128)
    for i in range (0,row):
        for j in range(0,col):
            d=np.power(i-row/2,2)+np.power(j-col/2,2)
            H[i,j]=np.exp(-(k*(np.power(d,5/6))))
    
    #Butterworth lowpass filter
    H_Butterworth=np.zeros([row,col],dtype=np.complex128)
    for i in range (0,row):
        for j in range(0,col):
            d=np.power(i-row/2,2)+np.power(j-col/2,2)
            H_Butterworth[i,j]=1/(1+np.power(d/(np.power(radius,2)),order))
    
    output_image=input_image_center*H_Butterworth/H

    output_image=np.fft.ifft2(np.fft.ifftshift(output_image))
    output_image=np.real(output_image)

    a=np.max(output_image)
    b=np.min(output_image)
    for i in range(0,output_image.shape[0]):
        for j in range(0,output_image.shape[1]):
            output_image[i,j]=int((output_image[i,j]-b)*255/(a-b))
    
    output_image=np.array(output_image,dtype=np.uint8)
    return output_image



def Wiener_filter(input_image,K):
    input_image_DFT=np.fft.fft2(input_image)
    input_image_center=np.fft.fftshift(input_image_DFT)

    row,col=input_image_center.shape

    H=np.zeros([row,col],dtype=np.complex128)
    for i in range (0,row):
        for j in range(0,col):
            d=np.power(i-row/2,2)+np.power(j-col/2,2)
            H[i,j]=np.exp(-(0.0025*(np.power(d,5/6))))
    
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

```

​	**Result and analysis:**

​		**Full inverse filter:** According to the information about `Q6_2.tif`, the value of $k$ is $0.0025$.

<img src="./Lab6 Image Restoration/image-20230503204208101.png" alt="image-20230503204208101" style="zoom: 50%;" />

<div align = 'center'><b>Fig.11 restored image using full inverse filter </div>

​		According to `Fig 11`, the performance of direct inverse filtering is very poor in general.

​	**Radially limited inverse filter:** The limit used in radially limited inverse filter is a Butterworth lowpass function of order $10$. The Butterworth Filters of order $n$ and with cutoff frequency $D_{0}$ is
$$
H(\mu,\nu)=\frac{1}{1+[\frac{D(\mu,\nu)}{D_{0}}]^{2n}}\\
D(\mu,\nu)=((\mu-\frac{M}{2})^{2}+(\nu-\frac{N}{2})^{2})^{\frac{1}{2}}
$$

<center class="half">    
    <img src="./Lab6 Image Restoration/image-20230503211338374.png" alt="image-20230503211338374" style="zoom:100%;"width="300"/>    
    <img src="./Lab6 Image Restoration/image-20230503211410649.png" alt="image-20230503211410649" style="zoom:100%;" width="300"/> 
</center

<center>radius=30 order=10 &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp radius=40 order=10 </center>

<center class="half">    
    <img src="./Lab6 Image Restoration/image-20230503205927705.png" alt="image-20230503205927705" style="zoom:100%;"width="300"/>    
    <img src="./Lab6 Image Restoration/image-20230503210011736.png" alt="image-20230503210011736" style="zoom:100%;" width="300"/> 
</center

<center>radius=50 order=10 &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp radius=60 order=10 </center>

<center class="half">    
    <img src="./Lab6 Image Restoration/image-20230503210034160.png" alt="image-20230503210034160" style="zoom:100%;"width="300"/>    
    <img src="./Lab6 Image Restoration/image-20230503210103442.png" alt="image-20230503210103442" style="zoom:100%;" width="300"/> 
</center

<center>radius=70 order=10 &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp radius=80 order=10 </center>

<div align = 'center'><b>Fig.12 restored image using radially limited inverse filter </div>

​	The restoration performance of radially limited inverse filter is better than that of full inverse filter because it mainly preserve the frequency component in the low frequency area where the effective signal mainly concentrated. Among the above figures,  the restoration performance with radius $60$ and order $10$ is the best  because it preserves the most edge information, making the picture more detailed.

<center class="half">    
    <img src="./Lab6 Image Restoration/image-20230503214318476.png" alt="image-20230503214318476" style="zoom:100%;"width="300"/>    
    <img src="./Lab6 Image Restoration/image-20230503214351354.png" alt="image-20230503214351354" style="zoom:100%;" width="300"/> 
</center

<center>radius=60 order=5 &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp radius=60 order=10 </center>

<center class="half">    
    <img src="./Lab6 Image Restoration/image-20230503214428696.png" alt="image-20230503214428696" style="zoom:100%;"width="300"/>    
    <img src="./Lab6 Image Restoration/image-20230503214455012.png" alt="image-20230503214455012" style="zoom:100%;" width="300"/> 
</center

<center>radius=60 order=15 &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp radius=60 order=20 </center>

​	From the above four pictures, the restoration performances with order $10$, order $15$ and order $20$ are almost the same when radius is $60$, which are better than that with order $5$. According to the theoretical theory, with order increasing, the filter is steeper at the boundary, which makes it more idealized. 

<img src="./Lab6 Image Restoration/image-20230503215039406.png" alt="image-20230503215039406" style="zoom:67%;" />

<div align = 'center'><b>Fig.13 Butterworth Lowpass filter </div>

​	**Wiener filter:** 

<center class="half">    
    <img src="./Lab6 Image Restoration/image-20230503220438078.png" alt="image-20230503220438078" style="zoom:100%;"width="300"/>    
    <img src="./Lab6 Image Restoration/image-20230503220524056.png" alt="image-20230503220524056" style="zoom:100%;" width="300"/> 
</center

<center>K=0.0025 &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp K=0.1 </center>

<center class="half">    
    <img src="./Lab6 Image Restoration/image-20230503220557372.png" alt="image-20230503220557372" style="zoom:100%;"width="300"/>    
    <img src="./Lab6 Image Restoration/image-20230503220625223.png" alt="image-20230503220625223" style="zoom:100%;" width="300"/> 
</center

<center>K=0.01 &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp K=0.001 </center>

<center class="half">    
    <img src="./Lab6 Image Restoration/image-20230503220659402.png" alt="image-20230503220659402" style="zoom:100%;"width="300"/>    
    <img src="./Lab6 Image Restoration/image-20230503220729797.png" alt="image-20230503220729797" style="zoom:100%;" width="300"/> 
</center

<center>K=0.0001 &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp K=0.00001 </center>

<div align = 'center'><b>Fig.13 restored image using Wiener filter </div>

​	From the above six figures, it is easy to be known that with $K$ decreasing, the restoration image is more and more clear. However, when $K$ is less or equal than $0.0001$, wave spots appear in the restoration image although the restoration image is much clear. So, among the above figures, the restoration performance with $K=0.001$ is the best.

​	**algorithm complexity:** 

|                | full inverse filter | radially limited inverse filter(radius 60 order 10) | wiener filter(K=0.001) |
| :------------: | :-----------------: | :-------------------------------------------------: | :--------------------: |
| average time/s |      1.5495721      |                      3.1425609                      |       1.5647462        |
|     space      |        f~5MN        |                        f~7MN                        |         f~6MN          |

​	It is obvious that the average time used by radially limited inverse filter with radius $60$ and order $10$ is longest among the three algorithm. That used by Wiener filter with $K=0.001$ is the second and that used by full inverse filter is the least. One of the reasons is that the method of radially limited inverse filter with radius $60$ and order $10$ need to generate Butterworth lowpass filter with order $10$, which makes the average time longer. For full inverse filter, $5MN$ represents the total space which is used to store `input_image_DFT`, `input_image_center`, `H`, the total number of `d` and `output_image`. For radially limited inverse filter, in addition to $5MN$ which is used to store `input_image_DFT`, `input_image_center`, `H`, the total number of `d` generated in the process of generating `H` and `output_image`, another $2MN$ is used to store `H_Butterworth_lowpass_filter` and the total number of `d` generated in the process of generating `H_Butterworth_lowpass_filter`. For Wiener filter, another $1MN$ is used to store `buf`, which is $|H(\mu,\nu)|^{2}$.

### 2.3 Image restoration further

​	In Wiener filter, the power spectra of undegraded image and noise must be known. Although a constant estimate is sometimes useful, it is not always suitable. Constrained least squares filtering just requires the mean and variance of the noise. And it is optimum for the particular image processed. 

​	The model of image degradation may be written in matrix form:
$$
g=Hf+\eta
$$
​	where $g$, $\eta$ and $f$ are vectors of size $MN×1$, and $H$ is a degradation matrix of size $MN×MN$. Directly manipulating the matrix to solve $f$ is impractical due to the size of the problem, but the matrix form does facilitate the derivation of the restoration techniques. Central of the method is to base optimality of restoration on a measure of smoothness, such as the second derivative of an image(the Laplacian). So the problem is formulated as:
$$
\min C=\sum^{M-1}_{x=0}\sum^{N-1}_{y=0}[\nabla^{2}f(x,y)]^{2}\\
$$
​																				subject to $||g-H\widehat f||^{2}=||\eta||^{2}$

​	The frequency domain solution to this optimization problem is given by
$$
\widehat F(\mu,\nu)=[\frac{H^{*}(\mu,\nu)}{|H(\mu,\nu)|^{2}+\gamma |P(\mu,\nu)|^{2}}]G(\mu,\nu)
$$
​	where, $\gamma$ is a parameter to adjust, and $P(\mu,\nu)$ is the Fourier transform of the Laplacian operator
$$
p(x,y)=\left[ \begin{matrix}
	0&		-1&		0\\
	-1&		4&		-1\\
	0&		-1&		0\\
	
\end{matrix} \right]
$$
​	According to the lecture notes, `Q6_3_1.tif`, `Q6_3_2.tif` and `Q6_3_3` have been blurred using the function in `Formula (17)` with $a=b=0.1$ and $T=1$.

​	**Pseudo code for constrained least squares filtering:**

```python
input_image_DFT=np.fft.fft2(input_image)
input_image_center=np.fft.fftshift(input_image_DFT)

according to formula to generate H(u,v)

pad laplacian operator

laplacian_operator_DFT=np.fft.fft2(pad_laplacian_operator)
    laplacian_operator_center=np.fft.fftshift(laplacian_operator_DFT)
mask=np.conj(H)/(np.conj(H)*H+gama*np.conj(laplacian_operator_center)*laplacian_operator_center)

output_image=input_image_center*mask output_image=np.fft.ifft2(np.fft.ifftshift(output_image))
output_image=np.real(output_image)
bormalize output_image
```

​	**Python code:**

```python
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

```

​	**Result and analysis:**

​	For `Q6_3_1.tif`, the background noise is quite small, so Wiener filter and constrained least squares filter is direct to use.

<center class="half">    
    <img src="./Lab6 Image Restoration/image-20230504040956103.png" alt="image-20230504040956103" style="zoom:100%;"width="300"/>    
    <img src="./Lab6 Image Restoration/image-20230504041032632.png" alt="image-20230504041032632" style="zoom:100%;" width="300"/> 
</center

<center>K=0.1 &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp K=0.01 </center>

<center class="half">    
    <img src="./Lab6 Image Restoration/image-20230504041058754.png" alt="image-20230504041058754" style="zoom:100%;"width="300"/>    
    <img src="./Lab6 Image Restoration/image-20230504041131027.png" alt="image-20230504041131027" style="zoom:100%;" width="300"/> 
</center

<center>K=0.001 &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp K=0.0001 </center>

<center class="half">    
    <img src="./Lab6 Image Restoration/image-20230504041200928.png" alt="image-20230504041200928" style="zoom:100%;"width="300"/>    
    <img src="./Lab6 Image Restoration/image-20230504041246458.png" alt="image-20230504041246458" style="zoom:100%;" width="300"/> 
</center

<center>K=0.00001 &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp K=0.000001 </center>

<div align = 'center'><b>Fig.14 restored image using Wiener filter </div>

​	From the results using Wiener filter, with $K$ increasing, the result is more and more clear and the letters in the background are fading away. However, when $K$ is greater than or equal to $0.00001$, the image's contrast between light and dark is constantly decreasing.

<center class="half">    
    <img src="./Lab6 Image Restoration/image-20230504043313515.png" alt="image-20230504043313515" style="zoom:100%;"width="300"/>    
    <img src="./Lab6 Image Restoration/image-20230504043335531.png" alt="image-20230504043335531" style="zoom:100%;" width="300"/> 
</center

<center>gamma=0.1 &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp gamma=0.01 </center>

<center class="half">    
    <img src="./Lab6 Image Restoration/image-20230504043358184.png" alt="image-20230504043358184" style="zoom:100%;"width="300"/>    
    <img src="./Lab6 Image Restoration/image-20230504043430234.png" alt="image-20230504043430234" style="zoom:100%;" width="300"/> 
</center

<center>gamma=0.001 &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp gamma=0.0001 </center>

<center class="half">    
    <img src="./Lab6 Image Restoration/image-20230504050351371.png" alt="image-20230504050351371" style="zoom:100%;"width="300"/>    
    <img src="./Lab6 Image Restoration/image-20230504043521021.png" alt="image-20230504043521021" style="zoom:100%;" width="300"/> 
</center

<center>K=0.00001 &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp K=0.000001 </center>

<div align = 'center'><b>Fig.15 restored image using constrained least squares filter </div>

​	From the results using constrained least squares filter, with $\gamma$ increasing, the result is more and more clear while the image's contrast between light and dark is constantly decreasing after $\gamma$ is greater than or equal to $0.0001$. When $K$ and $\gamma$ are greater than $0.00001$, the result using Wiener filter is better than that using constrained least squares filter. When $K$ and  $\gamma$ are equal to $0.000001$, in my opinion, the result using Wiener filter is poorer than that using constrained least squares filter because the latter's contrast between light and dark is higher.

​	For `Q6_3_2.tif` and `Q6_3_3.tif`, the background noise is relatively large. Before applying the Wiener filter and constrained least squares filter, removing noise is an necessary operation.

​	For `Q6_3_2.tif`, adaptive median filter with $S_{max}=7$ and alpha-trimmed mean filter with size $5×5$ and $d=10$ are used to filter `Q6_3_2.tif`. The results are shown in `Fig 16`.

<center class="half">    
    <img src="./Lab6 Image Restoration/image-20230504184607967.png" alt="image-20230504184607967" style="zoom:100%;"width="300"/>    
    <img src="./Lab6 Image Restoration/image-20230504184635627.png" alt="image-20230504184635627" style="zoom:100%;" width="300"/> 
</center

<center>adaptive median filter &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp alpha-trimmed mean filter </center>

<div align = 'center'><b>Fig.16 Q6_3_2 after denoising </div>

​	From the results after removing noise, it is obvious that the performance of alpha-trimmed mean filter with size $5×5$ and $d=10$ is better than that of adaptive median filter with $S_{max}=7$ because the former is more effective in removing noise compared to the latter. So subsequent Wiener filtering and constrained least squares filtering are applied to the image filtered by alpha-trimmed filter with size $5×5$ and $d=10$.

 <center class="half">    
    <img src="./Lab6 Image Restoration/image-20230504184701634.png" alt="image-20230504184701634" style="zoom:100%;"width="300"/>    
    <img src="./Lab6 Image Restoration/image-20230504184720266.png" alt="image-20230504184720266" style="zoom:100%;" width="300"/> 
</center


<center>K=0.1 &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp K=0.01 </center>

 <center class="half">    
    <img src="./Lab6 Image Restoration/image-20230504184745461.png" alt="image-20230504184745461" style="zoom:100%;"width="300"/>    
    <img src="./Lab6 Image Restoration/image-20230504184805480.png" alt="image-20230504184805480" style="zoom:100%;" width="300"/> 
</center


<center>K=0.001 &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp K=0.0001 </center>

 <center class="half">    
    <img src="./Lab6 Image Restoration/image-20230504184830820.png" alt="image-20230504184830820" style="zoom:100%;"width="300"/>    
    <img src="./Lab6 Image Restoration/image-20230504184846553.png" alt="image-20230504184846553" style="zoom:100%;" width="300"/> 
</center


<center>K=0.00001 &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp K=0.000001 </center>

<div align = 'center'><b>Fig.17 results using Wiener filter of different K </div>

 <center class="half">    
    <img src="./Lab6 Image Restoration/image-20230504184910706.png" alt="image-20230504184910706" style="zoom:100%;"width="300"/>    
    <img src="./Lab6 Image Restoration/image-20230504184933935.png" alt="image-20230504184933935" style="zoom:100%;" width="300"/> 
</center

<center>gamma=0.1 &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp gamma=0.01 </center>

 <center class="half">    
    <img src="./Lab6 Image Restoration/image-20230504185000243.png" alt="image-20230504185000243" style="zoom:100%;"width="300"/>    
    <img src="./Lab6 Image Restoration/image-20230504185022212.png" alt="image-20230504185022212" style="zoom:100%;" width="300"/> 
</center

<center>gamma=0.001 &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp gamma=0.0001 </center>

 <center class="half">    
    <img src="./Lab6 Image Restoration/image-20230504185049780.png" alt="image-20230504185049780" style="zoom:100%;"width="300"/>    
    <img src="./Lab6 Image Restoration/image-20230504185112263.png" alt="image-20230504185112263" style="zoom:100%;" width="300"/> 
</center

<center>gamma=0.00001 &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp gamma=0.000001 </center>

<div align = 'center'><b>Fig.18 results using constraint least square filter of different gamma</div>

​	For using Wiener filter, the performance of Wiener filter with $K=0.001$ is the best among all of Wiener filters because it preserves the edge information to make the image clearer and also preserves the contrast of the original image better. When $K$ is less than $0.001$, the filtered image isn't enough clear. When $K$ is greater than $0.001$, the contrast of the original image is much distorted. For using constraint least square filter, the performance of constraint least square filter with $\gamma=0.00001$ is the best among all of constraint least square filter. When $\gamma$ is less than $0.00001$, the filtered image isn't enough clear. When $\gamma$ is greater than $0.00001$, the contrast of the original image is much distorted. What's more, the performance of constraint least square filter with $\gamma=0.00001$ is worse than that of Wiener filter with $K=0.001$ in terms of contrast.

​	For `Q6_3_2.tif`, adaptive median filter with $S_{max}=7$, alpha-trimmed mean filter with size $5×5$ and $d=10$, median filter with size $3×3$ and midpoint filter with size $3×3$ followed by adaptive median filter with $S_{max}=7$ are used to filter `Q6_3_2.tif`. The results are shown in `Fig 19`. 

<center class="half">    
    <img src="./Lab6 Image Restoration/image-20230504192822460.png" alt="image-20230504192822460" style="zoom:100%;"width="300"/>    
    <img src="./Lab6 Image Restoration/image-20230504192911251.png" alt="image-20230504192911251" style="zoom:100%;" width="300"/> 
</center

<center>adaptive median filter &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp alpha-trimmed mean filter </center>

<center class="half">    
    <img src="./Lab6 Image Restoration/image-20230504193241297.png" alt="image-20230504193241297" style="zoom:100%;"width="300"/>    
    <img src="./Lab6 Image Restoration/image-20230504193300739.png" alt="image-20230504193300739" style="zoom:100%;" width="300"/> 
</center

<center> median filter &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp midpoint filter followed by adaptive median filter </center>

<div align = 'center'><b>Fig.19 Q6_3_3 after denoising </div>

​	Among the above four results, it is obvious that the performances of alpha-trimmed mean filter with size $5×5$ and $d=10$  and median filter with size $3×3$ are better. Among them, although the result with median filter with size $3×3$ is a litter clearer than that with alpha-trimmed mean filter with size $5×5$ and $d=10$, there is still a relatively obvious salt-and-pepper noise in the image. So subsequent Wiener filtering and constrained least squares filtering are applied to the image filtered by alpha-trimmed filter with size $5×5$ and $d=10$.

 <center class="half">    
    <img src="./Lab6 Image Restoration/image-20230504194707183.png" alt="image-20230504194707183" style="zoom:100%;"width="300"/>    
    <img src="./Lab6 Image Restoration/image-20230504194747013.png" alt="image-20230504194747013" style="zoom:100%;" width="300"/> 
</center


<center>K=0.1 &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp K=0.01 </center>

 <center class="half">    
    <img src="./Lab6 Image Restoration/image-20230504194806290.png" alt="image-20230504194806290" style="zoom:100%;"width="300"/>    
    <img src="./Lab6 Image Restoration/image-20230504194829655.png" alt="image-20230504194829655" style="zoom:100%;" width="300"/> 
</center

<center>K=0.001 &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp K=0.0001 </center>

<center class="half">    
    <img src="./Lab6 Image Restoration/image-20230504194849519.png" alt="image-20230504194849519" style="zoom:100%;"width="300"/>    
    <img src="./Lab6 Image Restoration/image-20230504194911757.png" alt="image-20230504194911757" style="zoom:100%;" width="300"/> 
</center

<center>K=0.00001 &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp K=0.000001 </center>

<div align = 'center'><b>Fig.20 results using Wiener filter of different K </div>

 <center class="half">    
    <img src="./Lab6 Image Restoration/image-20230504194941067.png" alt="image-20230504194941067" style="zoom:100%;"width="300"/>    
    <img src="./Lab6 Image Restoration/image-20230504195015391.png" alt="image-20230504195015391" style="zoom:100%;" width="300"/> 
</center

<center>gamma=0.1 &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp gamma=0.01 </center>

 <center class="half">    
    <img src="./Lab6 Image Restoration/image-20230504195039389.png" alt="image-20230504195039389" style="zoom:100%;"width="300"/>    
    <img src="./Lab6 Image Restoration/image-20230504195112673.png" alt="image-20230504195112673" style="zoom:100%;" width="300"/> 
</center

<center>gamma=0.001 &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp gamma=0.0001 </center>

 <center class="half">    
    <img src="./Lab6 Image Restoration/image-20230504195150505.png" alt="image-20230504195150505" style="zoom:100%;"width="300"/>    
    <img src="./Lab6 Image Restoration/image-20230504195218191.png" alt="image-20230504195218191" style="zoom:100%;" width="300"/> 
</center

<center>gamma=0.00001 &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp gamma=0.000001 </center>

<div align = 'center'><b>Fig.21 results using constraint least square filter of different gamma</div>

​	For using Wiener filter, the performance of Wiener filter with $K=0.01$ is the best among all of Wiener filters because the edge information and contrast are well preserved, which makes the filtered image the most clear among these filtered images. When $K$ continues to decrease, the performance is worse and worse because Wiener filter is more and more closer to full inverse filter according to `formula (18)`. For using constraint least square filter, the performance of constraint least square filter with $\gamma=0.0001$ is the best among all of constraint least square filter. Although the contrast of the original image is much distorted, it preserves more edge information then other filtered images. However, the overall effect of constraint least square filter is worse than that of Wiener filter.

​	**Algorithm complexity:**

|              | Wiener filter(K=0.0001) | constrained least squares filter(gamma=0.0001) |
| :----------: | ----------------------- | ---------------------------------------------- |
| average time | 1.845777                | 1.8769261999999998                             |
|    space     | f~6MN                   | f~11MN                                         |

​	It is obvious that the average time used by Wiener filter with $K=0.0001$ is shorter than that used by constrained least squares filter with $\gamma=0.0001$. One of the reasons is that the method of constrained least squares filter with $\gamma=0.0001$ need to generate Laplacian operator, pad Laplacian operator, do Fourier transform of Laplacian operator and  center its transform. For Winer filter, $5MN$ represents the total space which is used to store `input_image_DFT`, `input_image_center`, `H`, the total number of `operator` and `output_image`. And $1MN$ is used to store `buf`, which is $|H(\mu,\nu)|^{2}$. For constrained least squares filter, another $5MN$ is used to store `pad_laplacian_operator`, `laplacian_operator_DFT`, `laplacian_operator_center`, `laplacian_operator_power` and `mask`. 

## 3.Conclusion

​	From the analysis about spatial domain filters and frequency domain filters, different types of filters have their own characteristic.

​	For contraharmonic mean filter, $Q>0$ for pepper noise and $Q<0$ for salt noise. Also, the performance of contraharmonic is related to $Q$. What's more, $Q$ and the size of local image determine the time complexity of it.

​	For a combination of salt-and-pepper and Gaussian noise, alpha-trimmed mean filter and a combination of adaptive mean filter and midpoint filter can be used. In this lab, the performance of alpha-trimmed mean filter with size $5×5$ and $d=10$ is worse than the combination of adaptive mean filter with $S_{max}=7$ and midpoint filter with size $3×3$. There is no doubt that the performance is related to the size. So, it cannot be concluded that the performance of alpha-trimmed mean filter is worse than the combination of adaptive mean filter and midpoint filter.

​	The radius and order of lowpass filter used in radially limited inverse filter can affect the performance of radially limited inverse filter. Theoretically, the higher the order, the better the performance. This is because the border of lowpass filter is steeper with higher order. But this cause the computation time increase. When the radius is so small, there's a lot of low-frequency components that get filtered out, which makes the filtered image blur. When the radius is large, the high frequency component cannot be sufficiently filtered out, which make the performance worse. In reality, it is necessary to select order and radius according to the actual situation.

​	In Wiener filter, the power spectra of the undegraded image and noise must be known. When them are unknown, a constant estimate is sometimes useful. According to the result of this experiment, when $K$ is relatively large, border information is not preserved enough. When $K$ is relatively small, Wiener filter is more closer to full inverse filter. The same is true for constraint least square filter. However, in this lab, the overall effect of constraint least square filter is worse than that of Wiener filter, which confuses me. I need to do more research to solve the puzzle.   
<h1 align = "center">Lab5 Filtering in the frequency domain</h1>

<center>张旭东 12011923</center>

## 1. Introduction

​	The frequency of an image is an indicator of the intensity of the change of gray in an image. The Fourier transform has a very obvious physical meaning in practice. In a mathematical sense, the Fourier transform transform the image from the spatial domain to the frequency domain, and  its inverse transform transforms the image from the frequency domain to the spatial domain. In other words, the physical meaning of Fourier transform is to transform the grayscale distribution function of an image to the frequency distribution function of an image. The significance of the light and dark points seen on the Fourier spectrum refers to the strength of difference between a point on the image and the point in the neighborhood, which is also called the magnitude of gradient. Generally speaking, the larger the magnitude of gradient, the stronger the brightness of the point. In this way, it is easy to know the distribution of energy of the image by observing the Fourier spectrum. If there are more dark points in the spectrum, the image in spatial domain is relatively soft, which means the difference between each point and the points in the neighborhood and the gradient is relative small. On the other hand, if there are many bright points in the spectrum, the image spectrum in the spatial domain is sharp with sharp boundaries and large difference of pixels on both sides of the boundary. 

​	The filters in frequency domain discussed in this experiment is Sobel filter and Butterworth notch reject filter.

​	Sobel filter is a kind of high-pass filter, which is also called directional filter because of the direction of it. It can remove the low frequency component in the image and keep the high frequency component in the image, making the the rate of change in the gradient direction more significant, which is used in the edge detection of an image.

​	Butterworth notch reject filter can be used to remove periodic noise, which means processing small region of the frequency rectangle. Although band-stop filter is also used to remove periodic noise, it attenuate components other than noise. The Butterworth notch reject filter mainly attenuates a certain point and does not have an effect to other components.

## 2. Analysis and result

### 2.1Theoretical knowledge

**2-D Discrete Fourier Transform and Its Inverse:**
$$
F(u,v)=\sum^{M-1}_{x=0}\sum^{N-1}_{y=0}f(x,y)e^{-j2\pi (\frac{ux}{M}+\frac{vy}{N})}\\
f(x,y)=\frac{1}{MN}\sum^{M-1}_{u=0}\sum^{N-1}_{v=0}F(u,v)e^{j2\pi (\frac{ux}{M}+\frac{vy}{N})}
$$
**Periodicity of the DFT:**

​	For $f(x,y)$ and its DFT $F(u,v)$, we have:
$$
f(x,y)e^{j2\pi (\frac{u_{0}x}{M}+\frac{v_{0}y}{N})} \Longleftrightarrow F(u-u_{0},v-v_{0})\\
f(x-x_{0},y-y_{0})\Longleftrightarrow F(u,v)e^{-j2\pi (\frac{ux_{0}}{M}+\frac{vy_{0}}{N})}
$$
​	2-D Fourier transform and its inverse are infinitely periodic,so
$$
F(u,v)=F(u+k_{1}M,v)=F(u,v+k_{2}N)=F(u+k_{1}M,v+k_{2}N)\\
f(x,y)=f(x+k_{1}M,y)=f(x,y+k_{2}N)=F(x+k_{1}M,y+k_{2}N)
$$
​	However, if we just take one period, an uncentered spectrum will be gotten. According to `Formula(2)`, let $u_{0}=\frac{M}{2},v_{0}=\frac{N}{2}$, then the spectrum will be moved to the center.

<center class="half">    
    <img src="./Lab5 Filtering in the frequency domain/image-20230403233800560.png" alt="image-20230403233800560" style="zoom:100%;"width="300"/>    
    <img src="./Lab5 Filtering in the frequency domain/image-20230403233822828.png" alt="image-20230403233822828" style="zoom:100%;" width="300"/> 
</center

<div align = 'center'><b>Fig.1 A 1-D DFT(left) and Shifted DFT obtained by multiplying f(x) by (-1)^x before computing F(u)(right)</div>

<center class="half">    
    <img src="./Lab5 Filtering in the frequency domain/image-20230403233842715.png" alt="image-20230403233842715" style="zoom:110%;"width="300"/>    
    <img src="./Lab5 Filtering in the frequency domain/image-20230403233901273.png" alt="image-20230403233901273" style="zoom:100%;" width="300"/> 
</center

<div align = 'center'><b>Fig.2 A 2-D DFT(left) and Shifted DFT obtained by multiplying f(x,y) by (-1)^(x+y) before computing F(u,v)(right)</div>

**Zero padding:**

​	As we all know, convolution operation in the spatial domain means product operation in the frequency domain. And product operation in the spatial domain means convolution operation in the frequency domain.
$$
f(x,y)*h(x,y)\Longleftrightarrow F(u,v)H(u,v)\\
f(x,y)h(x,y)\Longleftrightarrow F(u,v)*H(u,v)
$$
​	However, the data from adjacent periods produce wraparound error, yielding an incorrect convolution result. To obtain the correct result, function padding must be used.

<img src="./Lab5 Filtering in the frequency domain/image-20230403235926517.png" alt="image-20230403235926517" style="zoom: 67%;" />

<div align = 'center'><b>Fig.3 convolution of two discrete functions</div>

​	Let $f(x,y)$ and $h(x,y)$ be two image arrays of sizes $A×B$ and $C×D$ pixels, respectively. Wraparound error in their convolution can be avoided by padding these functions with zeros.
$$
f_{p}(x,y)=   \left\{ \begin{array}{rcl} f(x,y) & 0\leqslant x \leqslant A-1 &and& 0\leqslant y \leqslant B-1  \\0 &   A\leqslant x \leqslant P &or& B\leqslant y \leqslant Q \\
 \end{array}\right.
$$

$$
h_{p}(x,y)=   \left\{ \begin{array}{rcl} h(x,y) & 0\leqslant x \leqslant C-1 &and& 0\leqslant y \leqslant D-1  \\0 &   C\leqslant x \leqslant P &or& D\leqslant y \leqslant Q \\
 \end{array}\right.
$$

$$
p\geqslant A+C-1\\
Q\geqslant B+D-1\\
$$

**Steps for Filtering in the Frequency Domain:** 	

1. Given an input image $f(x,y)$ of size $M×N$， obtain the padding parameters $P$ and $Q$. Typically, $P=2M$ and $Q=2N$.

2. Form a padded image, $f_{p}(x,y)$ of size $P×Q$ by appending the necessary number of zeros to $f(x,y)$

3. Multiply $f_{p}(x,y)$ to center its transform

4. Compute the DFT,$F(u,v)$ of the image from the step $3$.

5. Generate a real, symmetric filter function, $H(u,v)$, of size $P×Q$ with center at coordinates $(P/2,Q/2)$

6. Form the product $G(u,v)=H(u,v)F(u,v)$ using array multiplication

7. Obtain the processed image
   $$
   g_{p}(x,y)=[real[\varsigma^{-1}[G(u,v)]]](-1)^{x+y}
   $$

8. Obtain the final processed result, $g(x,y)$, by extracting the $M×N$ region from the top,left quadrant of $g_{p}(x,y)$ 

### 2.2 Sobel filter

**principle:** Sobel operator is a discrete differential operator, which is used to calculate the approximate gradient of image gray function. That is, Sobel operator can be used to measure the change of image in vertical and horizontal directions. Due to the discrete characteristics of pixels, Sobel operator provides the approximation of image gradient through pixel difference in horizontal and vertical directions.

**Question formulation:**

​	The Sobel operator is as below:

<img src="./Lab5 Filtering in the frequency domain/image-20230404011328945.png" alt="image-20230404011328945" style="zoom:200%;" />

<div align = 'center'><b>Fig.4 spatial mask of Sobel</div>

<center class="half">    
    <img src="./Lab5 Filtering in the frequency domain/image-20230404012114264.png" alt="image-20230404012114264" style="zoom:100%;"width="300"/>    
    <img src="./Lab5 Filtering in the frequency domain/image-20230404012154290.png" alt="image-20230404012154290" style="zoom:100%;" width="300"/> 
</center

<div align = 'center'><b>Fig.5 perspective plot of its corresponding frequency domain(left) and filter shown as an image(right)</div>

​	The convolution of a filter $w(x,y)$ of size $m×n$ with an image $f(x,y)$ with $M×N$, denoted $w(x,y)\bigotimes f(x,y)$
$$
w(x,y)\bigotimes f(x,y)=\sum^{a}_{s=-a}\sum^{b}_{t=-b}w(s,t)f(x-s,y-t)
$$
​	After simplification,
$$
w(x,y)\bigotimes f(x,y)=G_{x}
$$
​	The operation in frequency domain is:
$$
O(u,v)=W(u,v)F(u,v)
$$
​	For generating from a given spatial filter, the steps of generating $H(u,v)$ is

1. ​	pad the image and the filter
   $$
   f_{p}(x,y)=   \left\{ \begin{array}{rcl} f(x,y) & 0\leqslant x \leqslant M-1 &and& 0\leqslant y \leqslant N-1  \\0 &   M\leqslant x \leqslant M+m-1 &or& N\leqslant y \leqslant N+n-1 \\
    \end{array}\right.
   $$

   $$
   h_{p}(x,y)=   \left\{ \begin{array}{rcl} h(x,y) & \frac{M}{2}\leqslant x \leqslant \frac{M}{2}+m-1 &and& \frac{N}{2}\leqslant y \leqslant \frac{N}{2}+n-1  \\0 &   others \\
    \end{array}\right.
   $$

2. multiply $h_{p}(x,y)$ by $(-1)^{x+y}$ to center the frequency domain filter

3. compute the forward DFT of the result in $(1)$

4. set the real part of the result DFT to $0$ to account for parasitic real parts

5. multiply the result by $(-1)^{u+v}$, which is implicit when $h(x,y)$ was moved to the center of $h_{p}(x,y)$.

**pseudo code:**

```python
#spatial
padimage=np.pad(input_image,((W,W),(H,H)),'symmetric')
for i in range (H,2*H):
    for j in range(W,2*W):
        output_image[i-H,j-W]=sobel_operator

output_image=normalize(output_image)

#frequency domain
padimage=np.zeros([input_image.shape[0]+mask_H-1,input_image.shape[1]+mask_W-1])
padimage[0:input_image.shape[0],0:input_image.shape[1]]=input_image
padimage=padimage*((-1)**(x+y))
padimage_DFT=FFT(padimage)

sobel=np.zeros([input_image.shape[0]+mask_H-1,input_image.shape[1]+mask_W-1])

sobel[input_image.shape[0]:input_image.shape[0]+mask_H,input_image.shape[1]:input_image.shape[1]+mask_W]=sobel_operator

sobel=sobel*((-1)**(x+y))
sobel_DFT=FFT(sobel)
sobel_DFT.real=0
sobel_DFT=sobel_DFT*((-1)**(u+v))

real_inverse_DFT=real(IFFT(padimage_DFT*sobel_DFT))*((-1)**(x+y))
output_image=real_inverse_DFT[0:input_image.shape[0],0:input_image.shape[1]]
output_image=normalize(output_image)

```

**Python code:**	

```python
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

    #pad the mask and DFT
    sobel_operator=[[-1,0,1],[-2,0,2],[-1,0,1]]

    pad_sobel_operator=np.zeros([P,Q],dtype=np.int32)
    pad_sobel_operator[int(1+(P-1-mask_H)/2):int(1+(P-1-mask_H)/2+mask_H),int(1+(Q-1-mask_W)/2):int(1+(Q-1-mask_W)/2+mask_W)]=sobel_operator

    for i in range (0,P):
        for j in range (0,Q):
            pad_sobel_operator[i,j]=pad_sobel_operator[i,j]*((-1)**(i+j))
    
    pad_sobel_operator_DFT=np.fft.fft2(pad_sobel_operator)
    pad_sobel_operator_DFT.real=0
    

    for u in range (0,pad_sobel_operator_DFT.shape[0]):
        for v in range (0,pad_sobel_operator_DFT.shape[1]):
            pad_sobel_operator_DFT[u,v]=pad_sobel_operator_DFT[u,v]*((-1)**(u+v))


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

```

**result:**

<center class="half">    
    <img src="./Lab5 Filtering in the frequency domain/image-20230404023013864.png" alt="image-20230404023013864" style="zoom:100%;"width="300"/>    
    <img src="./Lab5 Filtering in the frequency domain/image-20230404022948697.png" alt="image-20230404022948697" style="zoom:100%;" width="300"/> 
</center

<div align = 'center'><b>Fig.6 result of filtering in spatial domain(left) and result of filtering in frequency domain(right)</div>

**Analysis:** From the above pictures, the sharpening effect of Sobel filter is apparent. The edge information of the object is better extracted. However, there is a little difference between them. The overall brightness of the image filtered in spatial domain is higher than the image filtered in frequency domain. I think two possible reasons behind this is the symmetric pad of the original image and Sobel operator added by $128$ in frequency domain.  

**why perform a shift in Step 4 on slide 79 of Lecture 4 in the first Exercise:**

​	let $F(u,v)$ denote the spectrum of $h(x,y)$ and  $F_{p}(u,v)$ denote the spectrum of $h_{p}(x,y)$, the relation between $h(x,y)$ and $h_{p}(x,y)$ and that between $F(u,v)$ and $F_{p}(u,v)$ is:
$$
h_{p}(x,y)=h(x-\frac{M}{2},y-\frac{N}{2})\Longleftrightarrow F_{p}(u,v)=F(u,v)e^{-j\pi (u+v)}
$$
​	Then 
$$
h_{p}(x,y)(-1)^{(x+y)}\Longleftrightarrow F_{p}(u,v)=F(u-\frac{M}{2},v-\frac{N}{2})e^{-j\pi (u+v)}
$$
​	The purpose of the shift in Step 4 is to make $F(u-\frac{M}{2},v-\frac{N}{2})e^{-j\pi (u+v)}$ become $F(u-\frac{M}{2},v-\frac{N}{2})$, which can eliminate frequency offset and ensure result we want at the top,left quadrant of IFFT of filtered result. What's more, this operation can keep the phase of image, which can store the profile information of the image.
$$
e^{-j\pi (u+v)}(-1)^{u+v}=1
$$
<img src="./Lab5 Filtering in the frequency domain/image-20230404031109292.png" alt="image-20230404031109292" style="zoom:50%;" />

<div align = 'center'><b>Fig.7 output without multiplying the result by $(-1)^{u+v}$</div>

**algorithm complexity:**

|                | spatial | Frequency |
| -------------- | :-----: | --------- |
| average time/s |  0.890  | 1.197     |

​	It is obvious that the average time used by spatial operation is less than that used by frequency domain. The main time in spatial domain is used in operation of Sobel operator, whose complexity of time is $MNT_{add}$.($T_{add}$ means the time one addition takes). The main time in frequency domain is used in multiply, FFT and IFFT, whose complexity of time is $(M+m)(N+n)T_{multiply}+(M+m)(N+n)log((M+m)(N+n))T_{multiply}$.($T_{multiply}$ means the time one multiply takes).

**Other processing:**

<center class="half">    
    <img src="./Lab5 Filtering in the frequency domain/image-20230404054133745.png" alt="image-20230404054133745" style="zoom:100%;"width="300"/>    
    <img src="./Lab5 Filtering in the frequency domain/image-20230404054239813.png" alt="image-20230404054239813" style="zoom:100%;" width="300"/> 
</center
<div align = 'center'><b>Fig.8 correlation using vertical Sobel operator in spatial domain(left) and correlation using horizontal and vertical Sobel operator in spatial domain(right)</div>

​	Comparing the two pictures with the result filtering in frequency domain, it is obvious that he first two methods give better results. The result of correlation using horizontal and vertical Sobel operator in spatial domain is the best among the three because it makes the changes in horizontal direction and vertical direction more noticeable.

### 2.3 Butterworth notch reject filter

**principle:** Butterworth notch reject filter processes small regions of the frequency rectangle. To keep the resultant image with real intensity values, the zero-phase-shift must be symmetric about the origin. A notch with center at $(u_{0},v_{0})$ must have a corresponding notch at location $(-u_{0},-v_{0})$. Notch reject filters are constructed as products of high-pass filter whose center have been translated to the centers of the notches.

**Question formulation:** According  to the principle, the formula of notch reject filters is：
$$
H_{NR}=\prod ^{Q}_{k=1}H_{k}(u,v)H_{-k}(u,v)
$$
​	where $H_{k}(u,v)$  and $H_{-k}(u,v)$ are high-pass filters whose centers are at $(u_{k},v_{k})$ and $(-u_{k},-v_{k})$,respectively.

​	A Butterworth notch reject filter of order $n$ is:
$$
H_{NR}=\prod ^{K}_{k=1}[\frac{1}{1+[\frac{D_{0k}}{D_{k}(u,v)}]^{2n}}][\frac{1}{1+[\frac{D_{0k}}{D_{-k}(u,v)}]^{2n}}]
$$
​	Where
$$
D_{k}(u,v)=[(u-M/2-u_{k})^{2}+(v-N/2-v_{k})^{2}]^{\frac{1}{2}}\\
D_{-k}(u,v)=[(u-M/2+u_{k})^{2}+(v-N/2+v_{k})^{2}]^{\frac{1}{2}}
$$
​	So, an important thing is to get the center coordinates of $H_{k}(u,v)$, which can be realized by `plt.imshow`.

 **pseudo code:**

```python
padimage=np.zeros([2*input_image.shape[0],2*input_image.shape[1])
padimage[0:input_image.shape[0],0:input_image.shape[1]]=input_image
padimage=padimage*((-1)**(x+y))
padimage_DFT=FFT(padimage)
                   
click the point in the spectrum of padimage to get the center coordinats
                   
Butterworth_notch_reject_filter=np.ones([2*input_image.shape[0],2*input_image.shape[1]]
Butterworth_notch_reject_filter=construct(coordinate,n,D0)
                                        
real_inverse_DFT=real(IFFT(padimage_DFT*Butterworth_notch_reject_filter))((-1)**(x+y))
                                        
output_image=real_inverse_DFT[0:input_image.shape[0],0:input_image.shape[1]]                                        
output_image=clip(output_image,0,255)                       
```

**python code:**

```python
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
    
```

**result:**

<center class="half">    
    <img src="./Lab5 Filtering in the frequency domain/image-20230404035921997.png" alt="image-20230404035921997" style="zoom:100%;"width="300"/>    
    <img src="./Lab5 Filtering in the frequency domain/image-20230404035952054.png" alt="image-20230404035952054" style="zoom:85%;" width="300"/> 
</center

<div align = 'center'><b>Fig.9 D0=3 n=4</div>

<center class="half">    
    <img src="./Lab5 Filtering in the frequency domain/image-20230404040059461-1680552061718-1.png" alt="image-20230404040059461" style="zoom:100%;"width="300"/>    
    <img src="./Lab5 Filtering in the frequency domain/image-20230404040259899.png" alt="image-20230404040259899" style="zoom:85%;" width="300"/> 
</center

<div align = 'center'><b>Fig.10 D0=5 n=4 </div>

<center class="half">    
    <img src="./Lab5 Filtering in the frequency domain/image-20230404040334679.png" alt="image-20230404040334679" style="zoom:100%;"width="300"/>    
    <img src="./Lab5 Filtering in the frequency domain/image-20230404040352390.png" alt="image-20230404040352390" style="zoom:85%;" width="300"/> 
</center

<div align = 'center'><b>Fig.11 D0=10 n = 4</div>

<center class="half">    
    <img src="./Lab5 Filtering in the frequency domain/image-20230404040437134.png" alt="image-20230404040437134" style="zoom:100%;"width="300"/>    
    <img src="./Lab5 Filtering in the frequency domain/image-20230404040506845.png" alt="image-20230404040506845" style="zoom:85%;" width="300"/> 
</center

<div align = 'center'><b>Fig.12 D0=20 n=4</div>

<center class="half">    
    <img src="./Lab5 Filtering in the frequency domain/image-20230404041222259.png" alt="image-20230404041222259" style="zoom:100%;"width="300"/>    
    <img src="./Lab5 Filtering in the frequency domain/image-20230404041243119.png" alt="image-20230404041243119" style="zoom:85%;" width="300"/> 
</center

<div align = 'center'><b>Fig.13 D0=3 n=2</div>

<center class="half">    
    <img src="./Lab5 Filtering in the frequency domain/image-20230404041412445.png" alt="image-20230404041412445" style="zoom:100%;"width="300"/>    
    <img src="./Lab5 Filtering in the frequency domain/image-20230404041438095.png" alt="image-20230404041438095" style="zoom:85%;" width="300"/> 
</center

<div align = 'center'><b>Fig.14 D0=3 n=6</div>

<center class="half">    
    <img src="./Lab5 Filtering in the frequency domain/image-20230404041522999.png" alt="image-20230404041522999" style="zoom:100%;"width="300"/>    
    <img src="./Lab5 Filtering in the frequency domain/image-20230404041602353.png" alt="image-20230404041602353" style="zoom:85%;" width="300"/> 
</center

<div align = 'center'><b>Fig.15 D0=3 n=8</div>

**Analysis:** From the above pictures, what can be known is that with $D0$ increasing in a certain range, Butterworth notch reject filter works better. However, the effect it attenuates components other than noise is stronger according to `Formula(18)`. With $n$ increasing, the fuzzy component around the notch point is decreasing. According to `Formula(18)`, with $n$ increasing, the closer points is to notch point, the faster the attenuation will be. 

**how the parameters in the notch filters are selected, and  why:** 

​	To determine the value of $K$ and the centers of high-pass filter, the function `plt.imshow` is used to the log spectrum. The value of $K$ is equal to half the number of noise points in the log spectrum we need to remove. And click the noise points in the log spectrum to get the coordinate of it, which can be used in `Formula(19)`.

<img src="./Lab5 Filtering in the frequency domain/image-20230404044226726.png" alt="image-20230404044226726" style="zoom:80%;" />

As for the value of $n$ and $D0$, they are determined by the size of the noise points in the log spectrum and the strength of difference between the noise points and the points in the neighborhood because $n$ and $D0$ have a great effect on the performance of removing the noise.

 **algorithm complexity:**

|     order      |   2   | 4     |   6   | 8     |
| :------------: | :---: | ----- | :---: | ----- |
| average time/s | 6.691 | 6.767 | 6.827 | 6.938 |

​	It is obvious that with $n$ increasing, the average time is increasing. The main time is used in in multiply, FFT and IFFT, whose complexity of time is $nM^{2}T_{multiply}+M^{2}logM^{2}T_{multiply}$.($T_{multiply}$ means the time one multiply takes).

## 2.4 Explain why 𝐻(μ, ν) has to be real and symmetric in the Step 5 on slide 69 of Lecture 4,  which is also the case for most of the filters used in this laboratory. However, there is an  exception. Explain the exception

​	The principle of filtering in frequency domain is that the spectrum of the image is multiplied by the spectrum of the filter and do IFFT to obtain the processed image. The DFT of an image can be expressed as :
$$
F(u,v)=R_{F}(u,v)+jI_{F}(u,v)
$$
The filtered expression in spatial domain is:
$$
g(x,y)=F^{-1}[H(u,v)R(u,v)+jH(u,v)I(u,v)]
$$
​	The the phase of the image stores the profile information of the image. Besides., $F(u,v)$ is periodic. These profile information can be kept when $H(u,v)$ is real and symmetric.

​	However, there is an exception, which is the $H(u,v)$ of the Sobel mask used in `Section 2.2`. The Sobel mask exhibits odd symmetry, provided that it is embedded in an array of zeros of even sizes. The odd symmetry is preserved with respect to the padded array in forming $h_{p}(x,y)$, $H_{p}(u,v)$ of $h_{p}(x,y)$ is purely imaginary and odd according to the symmetry properties of the 2-D DFT, which yields results that are identical to filtering the image spatially using $h(x,y)$. To keep the results identical, $H_{p}(u,v)$ has to be purely imaginary. Because of that, we need to set the real part of the result DFT of $h_{p}(x,y)(-1)^{x+y}$ to $0$. 

## 3. Conclusion

​	From the analysis about the two methods of filtering in frequency domain, The two methods have their own characteristics. 

​	Sobel operator is a discrete differential operator, which is used to calculate the approximate gradient of image gray function. That is, Sobel operator can be used to measure the change of image in vertical and horizontal directions. Due to the discrete characteristics of pixels, Sobel operator provides the approximation of image gradient through pixel difference in horizontal and vertical directions, which is used in the edge detection of an image.

​	Butterworth notch reject filter processes small regions of the frequency rectangle, which can be used to remove periodic noise. The Butterworth notch reject filter mainly attenuates a certain point and does not have an effect to other components.

​	What's more, to avoid the data from adjacent periods produce wraparound error, yielding an incorrect convolution result, function padding must be used.


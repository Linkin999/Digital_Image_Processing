# SUSTech_EE326_lab6

*Topic: Image Restoration*

*Author: 11911521钟新宇*

*Project: lab6 report for Digital Image Processing*

## Introduction

The simple image restore process can be modeled as a linear system of degradation and recovery. This is shown below.

<img src="https://typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.com/typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.comimage-20220424225203143.png" alt="image-20220424225203143" style="zoom:33%;" />

<center>Fig 1: A model of the image degradation / restoration process.

where $f(x,y)$ represents the clean image. $f(x,y)$ is passed through a degenerate system with a frequency response of $H$ , and superimposed with additive noise $n(x,y)$ to obtain the image $g(x,y)$ that we actually see. The purpose of image reconstruction is to remove the interference of noise $n(x,y)$ and then to obtain the estimated image $\hat{f}(x,y)$ by recovering the filter. 

The relationship in spatial domain is as below.
$$
g(x, y)=h(x, y) \star f(x, y)+\eta(x, y)
$$
And the relationship in frequency domain is as below.
$$
G(u, v)=H(u, v) F(u, v)+N(u, v)
$$
In this lab, we will process images containing only noise and no degradation in task 1 and degraded images without noise in task 2. In task 3, we will process degenerate images with additive white noise. Since the codes are too long, I put them in the appendix file named *SUSTech_EE326_lab6_Appendix.pdf*.

## Task 1: Remove the noise



The principal sources of noise in digital images arise during image acquisition and/or transmission. The performance of imaging sensors is affected by a
variety of factors, such as environmental conditions during image acquisition,
and by the quality of the sensing elements themselves. For instance, in acquiring images with a CCD camera, light levels and sensor temperature are
major factors affecting the amount of noise in the resulting image. Images are
corrupted during transmission principally due to interference in the channel
used for transmission. For example, an image transmitted using a wireless
network might be corrupted as a result of lightning or other atmospheric
disturbance.

In this section, we will review the types of white noise learned and the filters used to remove the noise.

### Noise Models

We use the probability density distribution (PDF) to distinguish different kinds of noise. The common noises are summarized in the following table.

| Noise                           | PDF                                                          | Histogram                                                    |
| ------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Gaussian Noise                  | <img src="https://typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.com/typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.comimage-20220424231322183.png" alt="image-20220424231322183" style="zoom:50%;" /> | <img src="https://typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.com/typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.comtypora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.comimage-20220424231241898.png" alt="image-20220424231241898" style="zoom: 33%;" /> |
| Rayleigh Noise                  | <img src="https://typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.com/typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.comimage-20220424231328054.png" alt="image-20220424231328054" style="zoom:50%;" /> | <img src="https://typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.com/typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.comtypora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.comimage-20220424231254447.png" alt="image-20220424231254447" style="zoom: 33%;" /> |
| Erlang (Gamma) Noise            | <img src="https://typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.com/typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.comimage-20220424231408335.png" alt="image-20220424231408335" style="zoom:50%;" /> | <img src="https://typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.com/typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.comimage-20220424231336548.png" alt="image-20220424231336548" style="zoom: 33%;" /> |
| Exponential Noise               | <img src="https://typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.com/typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.comimage-20220424231416842.png" alt="image-20220424231416842" style="zoom:50%;" /> | <img src="https://typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.com/typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.comimage-20220424231342934.png" alt="image-20220424231342934" style="zoom: 33%;" /> |
| Uniform Noise                   | <img src="https://typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.com/typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.comimage-20220424231428356.png" alt="image-20220424231428356" style="zoom:50%;" /> | <img src="https://typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.com/typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.comimage-20220424231352143.png" alt="image-20220424231352143" style="zoom: 33%;" /> |
| Impulse (Salt-and-Pepper) Noise | <img src="https://typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.com/typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.comimage-20220424231437863.png" alt="image-20220424231437863" style="zoom:50%;" /> | <img src="https://typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.com/typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.comimage-20220424231358817.png" alt="image-20220424231358817" style="zoom: 33%;" /> |

In the image, we can select a region with a uniform intensity distribution and draw a histogram of this region, then we can observe what noise is contained in the image.

| Exponential                                                  | Uniform                                                      | Salt & Pepper                                                |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| <img src="https://typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.com/typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.comimage-20220424231805186.png" alt="image-20220424231805186" style="zoom:33%;" /> | <img src="https://typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.com/typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.comimage-20220424231812888.png" alt="image-20220424231812888" style="zoom:33%;" /> | <img src="https://typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.com/typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.comimage-20220424231819000.png" alt="image-20220424231819000" style="zoom:33%;" /> |
| <img src="https://typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.com/typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.comimage-20220424231826480.png" alt="image-20220424231826480" style="zoom:33%;" /> | <img src="https://typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.com/typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.comimage-20220424231833571.png" alt="image-20220424231833571" style="zoom:33%;" /> | <img src="https://typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.com/typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.comimage-20220424231842038.png" alt="image-20220424231842038" style="zoom:33%;" /> |

### Nosie Reducing Filter

The researchers designed a variety of air domain filters and frequency domain filters to eliminate the noise. The common filters are shown below.

- Mean Filters

  - Arithmetic mean filter

    - The arithmetic averaging filter sums all intensity values in the neighborhood and finally averages them.
      $$
      \hat{f}(x, y)=\frac{1}{m n} \sum_{(s, t) \in S_{x y}} g(s, t)
      $$

  - Geometric mean filter  

    - The geometric average filter accumulates all intensity values in the neighborhood and then performs an exponential operation. In general, the geometric mean filter has a smoothing effect comparable to the arithmetic mean filter, but it loses less image detail. Besides, if the dynamic range of the image is large, we usually do logarithmic operation, but the geometric mean filter is usually not used after the logarithmic operation.
      $$
      \hat{f}(x, y)=\left[\prod_{(s, t) \in S_{x y}} g(s, t)\right]^{\frac{1}{m n}}
      $$

  - Harmonic mean filter

    - Harmonic mean filter works well for salt noise, but not for pepper noise. It also works well for other types of noise such as Gaussian noise.
      $$
      \hat{f}(x, y)=\frac{m n}{\sum_{(s, t) \in S_{x y}} \frac{1}{g(s, t)}}
      $$

  - Contraharmonic mean filter

    - Contraharmonic mean filter is good for reducing the effect of pepper or salt noise, but it can only handle one kind of noise at a time. q>0 handles pepper noise and q<0 handles salt noise. In addition, it destroys image details: Q>0 causes black areas to shrink and white areas to enlarge; Q<0 causes white areas to shrink and black areas to enlarge.
      $$
      \hat{f}(x, y)=\frac{\sum_{(s, t) \in S_{x y}} g(s, t)^{Q+1}}{\sum_{(s, t) \in S_{x y}} g(s, t)^{Q}}
      $$

- Order-Statistic Filters

  Order-Statistic Filter is a spatial filter whose response is based on ordering (sorting) the pixel values contained in the image area enclosed by the filter. The sorting result determines the response of the filter.

  - Median filter

    - The median filter updates the centroid intensity value using the median of the intensity values in the neighborhood. It is particularly effective against pretzel noise and does not cause blurring of the image edges or change the image shape size. But it is not effective against uniform noise.
      $$
      \hat{f}(x, y)=\operatorname{median}_{(s, t) \in S_{x y}}\{g(s, t)\}
      $$

  - Max filter

    - The max filter updates the centroid intensity value using the maximum of the intensity values in the neighborhood. It is suitable for handling pepper noise, but it causes the black areas of the image to become smaller and the white areas to become larger
      $$
      \hat{f}(x, y)=\max _{(s, t) \in S_{x y}}\{g(s, t)\}
      $$

  - Min filter

    - The min filter updates the centroid intensity value using the minimum of the intensity values in the neighborhood. It is suitable for handling salt noise, but it causes the white areas of the image to become smaller and the black areas to become larger
      $$
      \hat{f}(x, y)=\min _{(s, t) \in S_{x y}}\{g(s, t)\}
      $$
      

  - Midpoint filter

    - Mid point filter updates the center point intensity value using the minimum and maximum values of the intensity values in the neighborhood. It combines the advantages of the max filter and the min filter to handle pretzel noise.
      $$
      \hat{f}(x, y)=\frac{1}{2}\left[\max _{(s, t) \in S_{x y}}\{g(s, t)\}+\min _{(s, t) \in S_{x y}}\{g(s, t)\}\right]
      $$

  - Alpha-trimmed mean filter

    - The alpha-trimmed mean filter removes d pixels with lower intensity values and d pixels with higher intensity values in the neighborhood, and then updates the intensity value of the centroid using the arithmetic mean of the remaining intensity values.
      $$
      \hat{f}(x, y)=\frac{1}{m n-d} \sum_{(s, t) \in S_{x y}} g_{r}(s, t)
      $$

- Adaptive Filters

  - Adaptive, local noise reduction filter

    - The adaptive mean filter is equivalent to a weighted average of the original image and the arithmetic mean filter result, with the weight determined by the noise variance and the neighborhood variance. Since the noise added image is equal to the original image plus noise, the neighborhood variance must be greater than or equal to the noise variance.
          1. The noise variance is equal to 0, and the original image is output.
          2. The neighborhood variance is equal to the noise variance, and the arithmetic mean is output.
          3. The neighborhood variance is greater than the noise variance, which means that the neighborhood contains the effective information of the image, and the output image should be close to the original image (if the noise variance is small, then close to the original image; if the noise variance is large, then close to the arithmetic mean).

    - 
      $$
      \hat{f}(x, y)=g(x, y)-\frac{\sigma_{\eta}^{2}}{\sigma_{L}^{2}}\left[g(x, y)-m_{L}\right]
      $$

  - Adaptive median filter

    - Adaptive median filter is suitable for pretzel noise, which ensures that the output value is not pulsed as much as possible. It has two states, as shown below.

    - 
      $$
      \begin{aligned}
      z_{\min } &=\text { minimum intensity value in } S_{x y} \\
      z_{\max } &=\text { maximum intensity value in } S_{x y} \\
      z_{\text {med }} &=\text { median of intensity values in } S_{x y} \\
      z_{x y} &=\text { intensity value at coordinates }(x, y) \\
      S_{\max } &=\text { maximum allowed size of } S_{x y} \\
      A 1 &=z_{\text {med }}-z_{\min } \\
      A 2 &=z_{\mathrm{med}}-z_{\mathrm{max}} \\
      B 1 &=z_{x y}-z_{\min } \\
      B 2 &=z_{x y}-z_{\max }
      \end{aligned}
      $$

    - We use two conditions to determine the states.

        1. a1 > 0 and a2 < 0: This condition determines if the median is a pulse by comparing the relationship between the median and the maximum and minimum values in the neighborhood. If the condition is satisfied, it is not an impulse and goes to State B. If it is an impulse, then the window size is increased. If the window is increased to the maximum and the median is still a pulse, then the median is output directly.
       2. b1 > 0 and b2 < 0: This condition determines whether the point being processed is a pulse by comparing the original pixel with the maximum and minimum values in the neighborhood. If the condition is satisfied, it means it is not a pulse, and the gray value of the original pixel point is output. If it is a pulse, the median value of the neighborhood (not a pulse) is output, which is equivalent to median filtering.

### Results

#### Image 1

We found that this image contains only pepper noise with a noise variance of 0.1, so we presume that the Contraharmonic mean  filter with Q greater than 0, median filter, and max filter have better filtering effects. The result is also the same.

| <img src="https://typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.com/typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.comimage-20220424235312193.png" alt="image-20220424235312193" style="zoom:33%;" /> | <img src="https://typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.com/typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.comimage-20220424235317351.png" alt="image-20220424235317351" style="zoom:33%;" /> |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| <img src="https://typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.com/typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.comimage-20220424235323180.png" alt="image-20220424235323180" style="zoom:33%;" /> | <img src="https://typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.com/typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.comimage-20220424235327004.png" alt="image-20220424235327004" style="zoom:33%;" /> |



#### Image 2

We find that this image contains only salt noise with a noise variance of 0.1, so we presume that the contraharmonic mean  filter with Q less than 0, median filter, and min filter have better filtering effects. The result is also the same.

| <img src="https://typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.com/typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.comimage-20220424235347182.png" alt="image-20220424235347182" style="zoom:33%;" /> | <img src="https://typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.com/typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.comimage-20220424235351610.png" alt="image-20220424235351610" style="zoom:33%;" /> |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| <img src="https://typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.com/typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.comimage-20220424235356562.png" alt="image-20220424235356562" style="zoom:33%;" /> | <img src="https://typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.com/typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.comimage-20220424235359955.png" alt="image-20220424235359955" style="zoom:33%;" /> |



#### Image 3

We find that this image contains both pretzel noise and noise variance of 0.25. Therefore, we speculate that the max / min filter, mid point filter, harmonic / contraharmonic mean  filter which can only handle the same kind of noise, will not work. The results are the same, only the median filter works best, and the adaptive filter works better.

| <img src="https://typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.com/typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.comimage-20220424235413738.png" alt="image-20220424235413738" style="zoom:33%;" /> | <img src="https://typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.com/typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.comimage-20220424235417757.png" alt="image-20220424235417757" style="zoom:33%;" /> |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| <img src="https://typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.com/typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.comimage-20220424235421925.png" alt="image-20220424235421925" style="zoom:33%;" /> | <img src="https://typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.com/typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.comimage-20220424235424763.png" alt="image-20220424235424763" style="zoom:33%;" /> |



#### Image 4

We find that this image contains both pretzel noise and uniform noise. The median filter, adaptive filter, and arithmetic average filter all work well in this case.

| <img src="https://typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.com/typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.comimage-20220424235437482.png" alt="image-20220424235437482" style="zoom:33%;" /> | <img src="https://typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.com/typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.comimage-20220424235440732.png" alt="image-20220424235440732" style="zoom:33%;" /> |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| <img src="https://typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.com/typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.comimage-20220424235444252.png" alt="image-20220424235444252" style="zoom:33%;" /> | <img src="https://typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.com/typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.comimage-20220424235446865.png" alt="image-20220424235446865" style="zoom:33%;" /> |

## Task 2: Restore the image

### Degradation function

According to the previously introduced image degradation/restore model, the relationship between the restored and degraded images in the frequency domain is shown below.
$$
H_{s}(u, v)=\frac{G_{s}(u, v)}{\hat{F}_{s}(u, v)}
$$
In order to recover the image, we need to design a restore filter, which is the inverse filter of the degradation filter, and its frequency response is the inverse of the frequency response of the degradation filter. Therefore, in order to design the restorefilter, we first need to obtain the frequency response $H$ of the degenerate filter. Fortunately, researchers have summarized the degenerate functions of atmospheric turbulence and uniform motion of the camera.

- Atmospheric turbulence
  $$
  H(u, v)=e^{-k\left(u^{2}+v^{2}\right)^{5 / 6}}
  $$

- Motion blurring
  $$
  H(u, v)=\frac{T}{\pi(u a+v b)} \sin [\pi(u a+v b)] e^{-j \pi(u a+v b)}
  $$

### Restore Filters

In this lab, we learn about full inverse filter, radius-limited inverse filter, and Wiener filter. They are all frequency domain filters.

- full inverse filter

  The full inverse filter is very simple, it assumes that the degraded image is not disturbed by noise, then the relationship between the frequency response of the restored image, the degraded image, and the degraded filter is shown below.
  $$
  \hat{F}_{s}(u, v)=\frac{G_{s}(u, v)}{H_{s}(u, v)}
  $$
  Therefore, the full inverse filter serves to transform these three matrices into the frequency domain, perform the division operation, and then transform back into the spatial domain.

- radius-limited inverse filter

  Based on the fully inverse filter, the radius-limited inverse filter puts a limit on the size of the frequency, which is equivalent to a series of a fully inverse filter and a low-pass filter.

- Wiener filter

  The Wiener filter is also known as the minimum mean square error filter. It minimizes the mean square error of the original and recovered images.
  The minimum of the error function is given in the frequency domain by the expression.
  $$
  \hat{F}(u,v)=\left[\frac{1}{H(u, v)} \frac{|H(u, v)|^{2}}{|H(u, v)|^{2}+K}\right] G(u, v)
  $$

​	where $K=S_{\eta}(u, v) / S_{f}(u, v)$ is a constant.

### Results

- The original image

  <img src="https://typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.com/typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.comimage-20220425002653948.png" alt="image-20220425002653948" style="zoom: 33%;" />

- full inverse filtering

  |                           k=0.0025                           |                           k=0.001                            |                          k=0.00025                           |                           k=0.0001                           |
  | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
  | ![image-20220425002705972](https://typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.com/typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.comimage-20220425002705972.png) | ![image-20220425002709756](https://typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.com/typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.comimage-20220425002709756.png) | ![image-20220425002712770](https://typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.com/typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.comimage-20220425002712770.png) | ![image-20220425002715864](https://typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.com/typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.comimage-20220425002715864.png) |

- radius-limited inverse filtering (k=0.00025)

  |                          radius=40                           |                          radius=80                           |                          radius=120                          |                          radius=160                          |
  | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
  | ![image-20220425002812266](https://typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.com/typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.comimage-20220425002812266.png) | ![image-20220425002816702](https://typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.com/typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.comimage-20220425002816702.png) | ![image-20220425002821901](https://typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.com/typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.comimage-20220425002821901.png) | ![image-20220425002828846](https://typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.com/typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.comimage-20220425002828846.png) |

- Wiener filtering (k=0.00025)

  |                           K=1e-20                            |                           K=1e-15                            |                           K=1e-10                            |                            K=1e-5                            |
  | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
  | ![image-20220425003022574](https://typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.com/typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.comimage-20220425003022574.png) | ![image-20220425003026118](https://typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.com/typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.comimage-20220425003026118.png) | ![image-20220425003029211](https://typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.com/typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.comimage-20220425003029211.png) | ![image-20220425003032102](https://typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.com/typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.comimage-20220425003032102.png) |

We can observe that the best result is obtained with the Wiener filter for k=0.00025 and K=1e-15.

## Summary

In this lab, I learn about common noise in digital image processing, and the various filters used to remove noises. I also learn about image restoration methods, and learn about image degradation functions and restore filters. 
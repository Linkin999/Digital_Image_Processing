

# DIP

[TOC]



## LSB

### encode

1. 读取图像
2. 获得图像大小
3. 将 (256, 256) 的水印转为  (256, 256, 8)  的二进制水印
4. 将  (256, 256, 8)  的 3D 数组转为 (256\*256\*8) 的 1D 数组
5. 对 1D 数组补零为 (1024\*1024) 的 1D 数组
6. 将 (1024\*1024) 1D 数组转为 2D 数组
7. baker 映射
8. 替换图像的lsb位
9. 合成加密后的载体图像

### decode

1. 读取图像
2. 获得图像大小 (1024, 1024)
3. 提取lsb位
4. baker 映射
5. 转为 1D 数组 (1024\*1024)
6. 提取 (256\*256\*8) 的 1D 数组
7. (256\*256\*8) 1D to  (256, 256, 8) 3D
8. 合成水印

### Result

| ![image-20220528145701344](https://typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.com/typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.comimage-20220528145701344.png) | ![image-20220528145737247](https://typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.com/typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.comimage-20220528145737247.png) | ![image-20220528150310176](https://typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.com/typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.comimage-20220528150310176.png) | ![image-20220528145717201](https://typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.com/typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.comimage-20220528145717201.png) | ![image-20220528145741633](https://typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.com/typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.comimage-20220528145741633.png) |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|                           Carrier                            |                          Watermark                           |                       Baker watermark                        |                       Embedded Carrier                       |                     Extracted watermark                      |



### Attack

|                                        |                       Embedded Carrier                       |                     Extracted watermark                      |
| :------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|     Pepper noise, probability 0.1      | <img src="https://typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.com/typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.comimage-20220528151219738.png" alt="image-20220528151219738" style="zoom: 25%;" /> | ![image-20220528151301845](https://typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.com/typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.comimage-20220528151301845.png) |
|   Gaussian noise, mean 10, sigma 255   | <img src="https://typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.com/typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.comimage-20220528151209610.png" alt="image-20220528151209610" style="zoom:25%;" /> | ![image-20220528151250523](https://typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.com/typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.comimage-20220528151250523.png) |
|        Random noise, number 100        | <img src="https://typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.com/typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.comimage-20220528151215034.png" alt="image-20220528151215034" style="zoom:25%;" /> | ![image-20220528151255842](https://typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.com/typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.comimage-20220528151255842.png) |
|         Histogram equalization         | <img src="https://typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.com/typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.comimage-20220528151159887.png" alt="image-20220528151159887" style="zoom:25%;" /> | ![](https://typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.com/typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.commark_histogram_equal.bmp) |
|  Mean filter, neighborhood size (3,3)  | <img src="https://typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.com/typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.comimage-20220528151145127.png" alt="image-20220528151145127" style="zoom:25%;" /> | ![image-20220528151228884](https://typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.com/typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.comimage-20220528151228884.png) |
| Median filter, neighborhood size (3,3) | <img src="https://typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.com/typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.comimage-20220528151152885.png" alt="image-20220528151152885" style="zoom:25%;" /> | ![image-20220528151233813](https://typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.com/typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.comimage-20220528151233813.png) |



## FFT

### Result

| ![image-20220528152536438](https://typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.com/typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.comimage-20220528152536438.png) | ![image-20220528152551196](https://typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.com/typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.comimage-20220528152551196.png) | ![image-20220528152546689](https://typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.com/typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.comimage-20220528152546689.png) | ![image-20220528152512829](https://typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.com/typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.comimage-20220528152512829.png) | ![image-20220528152524043](https://typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.com/typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.comimage-20220528152524043.png) | ![image-20220528152645412](https://typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.com/typora-sadcbdsdnlsvnsnvlnvdnb.oss-cn-shenzhen.aliyuncs.comimage-20220528152645412.png) |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|                           Carrier                            |                          Watermark                           |                 Carrier in frequency domain                  |                          Embedding                           |                       Embedded Carrier                       |                      Decoded watermark                       |


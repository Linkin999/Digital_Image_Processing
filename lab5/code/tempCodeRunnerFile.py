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

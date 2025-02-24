'''
In CV2 Template matching, input images do not have to be of the same size. 
However in FFT NCC calculation, the input images have to be of the same sizes since the cross-correlation used here is operated in the frequency domain, 
where element wise multiplication is performed for the compelx-valued matrices, and produces the same matrices of the same dimension.

Observations
1. The simple NCC calculation method returns the same value as FFT NCC calculations, but not with our method of NCC calculation
    - This is because our method of NCC calculation circularly pads the main image (image 1), but not the template image (image 2)
2. Circularly padding both images and using FFT NCC calculations on the whole image results in same value but
3. Getting the center template original image to match with the padded image results in different value
4. Similarly, removing the circularly padded main image in our method of NCC calculation leads to different value

Modification: 
1. Added division by template area (M1 * M2) to account for local mean subtraction and to correctly scale variance terms to match cv2 matchtemplate NCC calculation 

However this results in very small deviation of NCC value from CV2 method, which should be fine since both ultimately uses very different method especially since FFT throws away some random coefficients
'''

import sys
import os
path = os.path.abspath("../VecRepV3") 
sys.path.append(path)
print(path)

from numpy.typing import NDArray
import numpy as np
from numpy import arange, zeros, int8, square, real, sqrt, exp
from numpy import absolute as abs
from numpy import complex64 as cmplx
from numpy import concatenate as cat
from numpy import conjugate as conj
from numpy import multiply as mult
from numpy.fft import fft2
from numpy.fft import ifft2
from math import pi
import matplotlib.pyplot as plt

import cv2
from  src.data_processing import ImageProducts 
############################# For testing ##############################
def normalized_cross_correlation(img1, img2):
    mean1, std1 = cv2.meanStdDev(img1)
    mean2, std2 = cv2.meanStdDev(img2)

    img1_norm = (img1 - mean1) / std1
    img2_norm = (img2 - mean2) / std2

    ncc = np.mean(img1_norm * img2_norm)
    
    return ncc

def get_NCC_score(input1, input2):
    scale = ImageProducts.scale_min(ImageProducts.ncc, -1)
    NCC_scaled_value = scale(input1, input2)
    return NCC_scaled_value
########################################################################
def find_max(A):
    i1, i2 = np.unravel_index(A.argmax(), A.shape)
    maximum = A[i1,i2]
    j1, j2 = np.unravel_index(A.argmin(), A.shape)
    minimum = A[j1,j2]
    return maximum, minimum, i1+1, i2+1

def template_functions(A1, kernel, N1, Q1, M1, P1, N2, Q2, M2, P2):
    fft_A1 = fft2(A1)
    squ_A1 = square(abs(A1))
    fft_squ_A1 = fft2(squ_A1)
    
    pg = zeros((N2,N1),dtype=int8)
    pg[0:M2,0:M1] = A1[Q2-1:Q2+M2-1,Q1-1:Q1+M1-1]
    
    IFTpg = ifft2(pg)*((N1*N2)/(M1*M2))
    
    tmp = ifft2(mult(fft_A1,kernel))
    gc = tmp[0:P2,0:P1]
    
    tmp = ifft2(mult(fft_squ_A1,kernel))
    gg = real(tmp[0:P2,0:P1])
    
    return gc, gg, IFTpg

##############################################################
def complex_ccor(A2, gc, gg, kernel, IFTpg,
                 N1, Q1, M1, P1, N2, Q2, M2, P2):
    fft_A2 = fft2(A2)
    squ_A2 = square(abs(A2))
    fft_squ_A2 = fft2(squ_A2)
    
    tmp = ifft2(mult(fft_A2,kernel))
    fc = tmp[0:P2,0:P1]
    
    tmp = ifft2(mult(kernel,fft_squ_A2))
    ff = real(tmp[0:P2,0:P1])
    
    tmp = ifft2(mult(fft_A2,IFTpg))
    fgc = tmp[0:P2,0:P1]
    
    gcq = gc[Q2-1,Q1-1]
    ggq = gg[Q2-1,Q1-1]

    numerator = real(fgc - (conj(fc) * gcq) / (M1 * M2)) 

    denominator_term1 = ff - (square(abs(fc)) / (M1 * M2))
    denominator_term2 = ggq - (square(abs(gcq)) / (M1 * M2))
    denominator = sqrt(denominator_term1 * denominator_term2)

    denominator[denominator <= 0] = 1e14 #tolerance
    
    return numerator/denominator

if __name__ == '__main__':

    data = np.load("data_1.npz")
    
    A1 = data["input1"]
    A2 = data["input2"]

    # =========================our ncc method
    value = get_NCC_score(A1,A2)   
    print(value)   

    original_dim = len(A1) #12
    # ====================== pure ncc comparison between 2 image
    ncc_result = normalized_cross_correlation(A1, A2)
    print(ncc_result)
    
    # Padding the main image with wrapped values to simulate wrapping
    A1 = np.pad(A1, max(len(A1), len(A1[0])), 'wrap')
    A2 = np.pad(A2, max(len(A2), len(A2[0])), 'wrap')  
    width = len(A2) 

    # 1 based indexing
    tx1 = 1 + original_dim 
    tx2 = 1 + original_dim + original_dim 
    
    ty1 = 1 + original_dim 
    ty2 = 1 + original_dim + original_dim
    
    n1 = width #36
    n2 = width  

    q1 = tx1  
    m1 = tx2 - tx1 + 1  
    p1 = n1-m1+1  
    
    q2 = ty1  
    m2 = ty2 - ty1 + 1  
    p2 = n2-m2+1  
    
    k1 = arange(1,n1)
    kernel1 = (1.0/m1)*((exp(1j*2*pi*m1*k1/n1) - 1)/(exp(1j*2*pi*k1/n1) - 1))
    kernel1 = cat(([1+1j*0.0], kernel1))
    
    k2 = arange(1,n2)
    kernel2 = (1.0/m2)*((exp(1j*2*pi*m2*k2/n2) - 1)/(exp(1j*2*pi*k2/n2) - 1))
    kernel2 = cat(([1+1j*0.0], kernel2))
    
    kernel = zeros((n2,n1),dtype=cmplx)
    for i1 in range(n1):
        for i2 in range(n2):
            kernel[i1][i2] = kernel2[i1]*kernel1[i2]  
    
    gc, gg, IFTpg = \
    template_functions(A1, kernel, n1, q1, m1, p1, n2, q2, m2, p2)
    
    cc = \
    complex_ccor(A2, gc, gg, kernel, IFTpg,
                 n1, q1, m1, p1, n2, q2, m2, p2)
    
    cc_max, cc_min, i2, i1 = find_max(cc)

    cc_max = cc_max*2 -1
    print(cc_max, i1, i2)

    # plt.figure(figsize=(10, 5))
    # plt.subplot(1, 2, 1)
    # plt.imshow(A1, cmap='gray') 
    # plt.subplot(1, 2, 2)
    # plt.imshow(A2, cmap='gray')
    # plt.savefig("output_images.png")
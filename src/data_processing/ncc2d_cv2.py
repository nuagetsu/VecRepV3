'''
This FFT calculation includes calculation of NCC Without Mean Subtraction, making it similar to how cv2.matchTemplate() uses TM_CCORR_NORMED.
However, this FFT calculation is unable to take input images of different dimensions yet. Hope to implement this in future work.
For a more standard NCC calculation that uses Mean Subtraction, refer to 'ncc2d.py' that uses TM_CCOEFF_NORMED.

Modification: 
1. Removed variables and functions involved in calculation of mean intensity.


Overall this results in an even smaller deviation of NCC value from CV2 method. For proof, please refer to README.
'''
import time
import sys
import os
path = os.path.abspath("../VecRepV3") 
sys.path.append(path)
print(path)

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

def find_max(A):
    i1, i2 = np.unravel_index(A.argmax(), A.shape)
    maximum = A[i1,i2]
    j1, j2 = np.unravel_index(A.argmin(), A.shape)
    minimum = A[j1,j2]
    return maximum, minimum, i1+1, i2+1

#obtain variables for template image 1 A1, function g
def template_functions(A1, kernel, N1, Q1, M1, P1, N2, Q2, M2, P2):
    fft_A1 = fft2(A1) #obtain beta which is in frequency domain from its data point A1
    squ_A1 = square(abs(A1))
    fft_squ_A1 = fft2(squ_A1) #obtain g hat square from its data point A1
   
    pg = zeros((N2,N1),dtype=int8)
    
    pg[0:M2,0:M1] = A1[Q2:Q2+M2,Q1:Q1+M1] #obtain the template window but zero padded

    FTpg = (fft2(pg)/ (N1*M1))/ (N2*M2) #inverse FFT of zero padded template image for h 
    
    tmp = ifft2(mult(fft_squ_A1,kernel)) #multiply fft of absolute squared data point of A1 with kernel then inverse FFT
    gg = real(tmp[0:P2,0:P1]) #g bar 
    
    return gg, FTpg

##############################################################

#obtain variables for main image 2 A2, function f
def complex_ccor(A2, gg, kernel, FTpg,
                 N1, Q1, M1, P1, N2, Q2, M2, P2):
    fft_A2 = fft2(A2) #obtain f hat which is in frequency domain from its data point A2
    squ_A2 = square(abs(A2))
    fft_squ_A2 = fft2(squ_A2) #obtain f hat square from its data point A2
    
    tmp = ifft2(mult(kernel,fft_squ_A2))
    ff = real(tmp[0:P2,0:P1]) #f hat squared

    tmp = fft2(mult(conj(fft_A2),FTpg))
    fgc = tmp[0:P2,0:P1]
    
    ggq = gg[Q2,Q1]

    numerator = real(fgc) 

    denominator_term1 = ff 
    denominator_term2 = ggq
    denominator = sqrt(denominator_term1 * denominator_term2)

    denominator[denominator <= 0] = 1e14 #tolerance
    
    return numerator/denominator

def ncc(A1, A2):
    original_dim = len(A1) #12
    
    # Padding the main image with wrapped values to simulate wrapping
    A1 = np.pad(A1, max(len(A1), len(A1[0])), 'wrap')
    A2 = np.pad(A2, max(len(A2), len(A2[0])), 'wrap')  
    width = len(A2) 

    tx1 = 0
    tx2 = original_dim
    
    ty1 = 0
    ty2 = original_dim
    
    n1 = width
    n2 = width  

    q1 = tx1  #start index on x axis of template 0
    m1 = tx2 - tx1  # length x-axis of template 12
    p1 = n1-m1+1  # total length of surroundings around template 24
    
    q2 = ty1  #start index on y axis of template 0
    m2 = ty2 - ty1 # length y-axis of template 12
    p2 = n2-m2+1  

    k1 = arange(1,n1)
    kernel1 = (1.0/m1)*((exp(1j*2*pi*m1*k1/n1) - 1)/(exp(1j*2*pi*k1/n1) - 1)) #gamma k
    kernel1 = cat(([1+1j*0.0], kernel1))
    
    k2 = arange(1,n2)
    kernel2 = (1.0/m2)*((exp(1j*2*pi*m2*k2/n2) - 1)/(exp(1j*2*pi*k2/n2) - 1)) #gamma k'
    kernel2 = cat(([1+1j*0.0], kernel2))
    
    kernel = zeros((n2,n1),dtype=cmplx)
    for i1 in range(n1):
        for i2 in range(n2):
            kernel[i1][i2] = kernel2[i1]*kernel1[i2]   #gamma k, k'
    
    gg, FTpg = \
    template_functions(A1, kernel, n1, q1, m1, p1, n2, q2, m2, p2)
    
    cc = \
    complex_ccor(A2, gg, kernel, FTpg,
                 n1, q1, m1, p1, n2, q2, m2, p2)
    
    cc_max, cc_min, i2, i1 = find_max(cc)

    cc_max = cc_max*2 -1
    return cc_max
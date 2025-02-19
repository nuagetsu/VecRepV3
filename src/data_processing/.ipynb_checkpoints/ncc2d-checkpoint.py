import os
import numpy as np
from numpy import arange
from numpy import zeros
from numpy import absolute as abs
from numpy import complex_ as cmplx
from numpy import int8
from numpy import square
from numpy import real
from numpy import sqrt
from numpy import exp
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

    numerator = real(fgc - conj(fc)*gcq)

    denominator = (ff-square(abs(fc)))* \
                (ggq-square(abs(gcq)))

    # denominator should be non-negative from the definition
    # of variances. It turns out that it takes negative values 
    # in the background where there is no tissue and the signal 
    # is dominated by noise. If this is the case we give it a 
    # large arbitrary value, therefore rendering the CC 
    # effectively zero at these points.

    denominator[denominator <= 0] = 1e14
    denominator = sqrt(denominator)

    return numerator/denominator

if __name__ == '__main__':

    tx1 = 308
    tx2 = 355

    n1 = 512
    q1 = tx1
    m1 = tx2-tx1+1
    p1 = n1-m1+1

    ty1 = 250
    ty2 = 303

    n2 = 512
    q2 = ty1
    m2 = ty2-ty1+1
    p2 = n2-m2+1

    A1 = np.fromfile("image1.dat",sep=" ").reshape(n2,n1)
    A2 = np.fromfile("image2.dat",sep=" ").reshape(n2,n1)

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

    print(cc_max, i1, i2)

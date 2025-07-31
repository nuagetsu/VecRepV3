'''
Distance functions to use as metrics for M-trees.
'''
import numpy as np
import math


import src.data_processing.ImageProducts as ImageProducts

def distance(img1, img2):
    ncc = min(ImageProducts.ncc(img1, img2), 1)
    return (math.acos(ncc) / (math.pi))

def getDistance(ncc_vector):
    ncc_vector[ncc_vector > 1] = 1
    ncc_vector[ncc_vector < 0] = 0
    return (np.arccos(ncc_vector) / (math.pi / 2))

def getDistanceMatrix(matrixG):
    matrixG[matrixG > 1] = 1
    matrixG[matrixG < 0] = 1
    return ((np.arccos(matrixG)) / (math.pi / 2))

def dist_fft_numba(img1, img2):
    ncc = min(ImageProducts.ncc_fft_numba(img1, img2), 1)
    return (math.acos(ncc) / (math.pi))

def dist_fft(img1, img2):
    ncc = min(ImageProducts.ncc_fft(img1, img2), 1)
    return (math.acos(ncc) / (math.pi))

def dist_indexed(item1, item2):
    img1 = item1[0]
    img2 = item2[0]
    return distance(img1, img2)

def dist_fft_indexed(item1, item2):
    img1 = item1[0]
    img2 = item2[0]
    return dist_fft(img1, img2)

def dist_fft_numba_indexed(item1, item2):
    img1 = item1[0]
    img2 = item2[0]
    return dist_fft_numba(img1, img2)
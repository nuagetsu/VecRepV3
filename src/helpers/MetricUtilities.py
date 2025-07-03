import matplotlib.pyplot as plt
import numpy as np
import math
import cmath
import pandas as pd


import src.data_processing.ImageProducts as ImageProducts

# rly get the octave working... rn it goes sliiightly out of 1 and sliightly out of 0 so even same image can be not 1 TT
def distance(img1, img2):
    #print(img1)
    #print(img2)
    #print(min(ImageProducts.ncc(img1, img2), 1))
    ncc = min(ImageProducts.ncc(img1, img2), 1)
    ncc = max(ncc, 0)
    #print(math.acos(ImageProducts.ncc(img1, img2)))
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
    ncc = max(ncc, 0)
    return (math.acos(ncc) / (math.pi / 2))

def dist_fft(img1, img2):
    ncc = min(ImageProducts.ncc_fft(img1, img2), 1)
    ncc = max(ncc, 0)
    return (math.acos(ncc) / (math.pi / 2))
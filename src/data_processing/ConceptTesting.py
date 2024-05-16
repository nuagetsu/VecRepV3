import cv2
import numpy as np
from numpy.typing import NDArray
from sklearn.preprocessing import normalize

"""
The purpose of this file is to test my understanding of concepts presented in the repo handed down to me from 
previous interns. I will probably use this as a test bed for things before implementing them in their own files.
In the future, I might convert this file into a worksheet of sorts to guide interns taking over.

Task 1: For all 2x2 matrices as outlined in the toy problem, find the Image Product Matrix G and then,
try to find the Embedding Matrix A.

"""

def produceG():
    img_set = np.array([[[0, 0], [0, 0]], [[0, 1], [0, 0]], [[0, 1], [0, 1]], [[0, 1], [1, 0]], [[0, 0], [1, 1]],
                        [[0, 1], [1, 1]], [[1, 1], [1, 1]]])
    G = []
    for mainImg in img_set:
        row = []
        for tempImg in img_set:
            row.append(ncc(mainImg, tempImg))
        G.append(row)
    return np.array(G)

def produceA():
    matrixG = produceG()
    eigenvalues, eigenvectors = np.linalg.eigh(matrixG)
    eigenvalues = eigenvalues[::-1]
    eigenvalues[0 > eigenvalues] = 0
    eigenvectors = eigenvectors.T[::-1].T
    Droot = np.sqrt(np.diag(eigenvalues))
    matrixA = np.matmul(Droot, eigenvectors.T)
    matrixA = normalize(matrixA, norm='l2', axis=0)
    return matrixA


# This NCC calculation is the same as the one in ImageProducts.py
def ncc(mainImg: NDArray, tempImg: NDArray):

    if np.sum(mainImg) == 0:
        if np.sum(tempImg) == 0:
            return 1
        return 0

    mainImg = np.pad(mainImg, max(len(mainImg), len(mainImg[0])), 'wrap')

    mainImg = np.asarray(mainImg, np.uint8)
    tempImg = np.asarray(tempImg, np.uint8)

    corr = cv2.matchTemplate(mainImg, tempImg, cv2.TM_CCORR_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(corr)

    return max_val

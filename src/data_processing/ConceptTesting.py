import math

import cv2
import random
import numpy as np
from numpy.typing import NDArray
from sklearn.preprocessing import normalize
from src.data_processing.EmbeddingFunctions import pencorr
from src.helpers.FindingEmbUsingSample import Lagrangian_Method2
import src.data_processing.ImageProducts as ip
import src.data_processing.ImageGenerators as ig
import src.visualization.Metrics as metrics
import pandas as pd
import matplotlib.pyplot as plt
from src.data_processing.EmbeddingFunctions import get_embedding_matrix
import src.data_processing.Utilities as utils
import src.helpers.FilepathUtils as fputils

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

def produceA(matrixG):
    eigenvalues, eigenvectors = np.linalg.eigh(matrixG)
    eigenvalues = eigenvalues[::-1]
    eigenvalues[0 > eigenvalues] = 0
    eigenvectors = eigenvectors.T[::-1].T
    Droot = np.sqrt(np.diag(eigenvalues))
    matrixA = np.matmul(Droot, eigenvectors.T)
    matrixA = normalize(matrixA, norm='l2', axis=0)
    return matrixA

"""
Task 2: Consider the following toy problem:

We wish to do similar image comparisons on triangles represented in a matrix. The triangles each fit into a 4x4 matrix
and are compared to an 8x8 main image which consists of another similar 4x4 triangle somewhere in the image (might be
in the middle).
"""
def triangleCalcs():
    smallestTriangle = [[1,0],
                        [1,1]]
    smallestTriangle = np.asarray(smallestTriangle, np.uint8)
    smallTriangle = [[0,0,0,0],
                     [0,0,0,0],
                     [1,0,0,0],
                     [1,1,0,0]]
    smallTriangle = np.asarray(smallTriangle, np.uint8)
    bigTriangle = [[1,0,0,0],
                   [1,1,0,0],
                   [1,1,1,0],
                   [1,1,1,1]]
    bigTriangle = np.asarray(bigTriangle, np.uint8)
    smallTriangleMultiplied = [[0,0,0,0,0,0,0,0],
                               [0,0,0,0,0,0,0,0],
                               [0,0,0,0,0,0,0,0],
                               [0,0,0,0,0,0,0,0],
                               [0,0,1,0,0,0,0,0],
                               [0,0,1,1,0,0,0,0],
                               [0,0,0,0,0,0,0,0],
                               [0,0,0,0,0,0,0,0]]
    smallTriangleMultiplied = np.asarray(smallTriangleMultiplied, np.uint8)
    bigTriangleMultiplied = [[0,0,0,0,0,0,0,0],
                             [0,0,0,0,0,0,0,0],
                             [0,0,1,0,0,0,0,0],
                             [0,0,1,1,0,0,0,0],
                             [0,0,1,1,1,0,0,0],
                             [0,0,1,1,1,1,0,0],
                             [0,0,0,0,0,0,0,0],
                             [0,0,0,0,0,0,0,0]]
    bigTriangleMultiplied = np.asarray(bigTriangleMultiplied, np.uint8)
    return (ncc(bigTriangleMultiplied, smallTriangle), ncc(smallTriangleMultiplied, bigTriangle),
            ncc(smallTriangleMultiplied, smallestTriangle), ncc(bigTriangleMultiplied, smallestTriangle))

"""
Findings from task 2:
If the triangle is indeed in the middle of the matrix, then there is a finite number of triangles that we have to
compare the template images to (either way, there will be a finite number, but the finite number is much smaller
for the case of the triangle being in the middle). This number is 64, so the dimension of our embedding matrix is at
most 64. Then, we can use our usual sampling method to create matrix G or G'. However, also note the problem of finding
smaller triangles.

Some smaller triangles can be found within larger triangles, as shown above, this is further exemplified if the main
image triangles are in the middle of the image. Then, when trying to match these smaller triangles, note that the NCC
score will be 1. So, we may be able to find new constraints for the sake of the Lagrangian method.

Since more comparisons will result in an NCC score of 1, it is possible that the training set may have to be larger to 
ensure that the G created will be more reliable. 

Subsequent task:
Consider this toy problem and create code/carry out previous procedures on this set of all 4x4 triangles.

From counting, there are 48x4=192 such triangles. This has been verified using combinatorics.
Next, we implement the image set. Use the rotate function to rotate every rotationally unique triangle to get every
triangle. Then, pad the surroundings with 2 rows of 0s.
"""
def generateImageSet(mean_subtracted=False, gridwide=False):
    unpaddedImageset = []
    two_by_two = np.array([[[1, 0],[1, 1]]])
    three_by_two = np.array([[[1, 0], [1, 0], [1, 1]], [[1, 0], [1, 1], [1, 0]], [[1, 1], [1, 0], [1, 0]],
                             [[1, 0], [1, 0], [0, 1]], [[0, 1], [1, 0], [1, 0]]])
    two_by_four = np.array([[[1, 1, 1, 1], [1, 0, 0, 0]], [[1, 1, 1, 1], [0, 1, 0, 0]], [[1, 1, 1, 1], [0, 0, 1, 0]],
                            [[1, 1, 1, 1], [0, 0, 0, 1]], [[0, 0, 1, 1], [1, 0, 0, 0]], [[1, 1, 0, 0], [0, 0, 0, 1]],
                            [[0, 1, 1, 1], [1, 0, 0, 0]], [[1, 1, 1, 0], [0, 0, 0, 1]]])
    three_by_three = np.array([[[1, 0, 0], [1, 1, 0], [1, 1, 1]], [[1, 0, 0], [1, 1, 1], [1, 0, 0]],
                               [[1, 0, 0], [1, 1, 0], [0, 0, 1]], [[0, 0, 1], [1, 1, 0], [0, 0, 1]],
                               [[0, 1, 0], [1, 1, 0], [0, 0, 1]]])
    three_by_four = np.array([[[1, 1, 1, 1], [1, 1, 0, 0], [1, 0, 0, 0]], [[1, 1, 1, 1], [0, 1, 1, 0], [0, 1, 0, 0]],
                              [[1, 1, 1, 1], [0, 1, 1, 0], [0, 0, 1, 0]], [[1, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 1]],
                              [[0, 0, 1, 0], [0, 1, 1, 1], [1, 0, 0, 0]], [[0, 0, 1, 0], [1, 1, 1, 0], [0, 0, 0, 1]],
                              [[1, 0, 0, 0], [1, 1, 1, 1], [1, 0, 0, 0]], [[0, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0]],
                              [[1, 0, 0, 0], [1, 1, 0, 0], [0, 0, 0, 1]], [[0, 0, 1, 1], [0, 1, 0, 0], [1, 0, 0, 0]],
                              [[0, 1, 0, 0], [1, 1, 1, 0], [0, 0, 0, 1]], [[0, 1, 0, 0], [0, 1, 1, 1], [1, 0, 0, 0]],
                              [[1, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], [[0, 1, 1, 1], [0, 1, 0, 0], [1, 0, 0, 0]],
                              [[1, 1, 1, 0], [0, 0, 1, 0], [0, 0, 0, 1]], [[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                              [[0, 0, 0, 1], [0, 1, 0, 0], [1, 0, 0, 0]]])
    four_by_four = np.array([[[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]],
                             [[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 1], [1, 0, 0, 0]],
                             [[1, 0, 0, 0], [1, 1, 1, 1], [1, 1, 0, 0], [1, 0, 0, 0]],
                             [[0, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                             [[0, 1, 0, 0], [0, 1, 0, 0], [1, 1, 1, 0], [0, 0, 0, 1]],
                             [[0, 0, 0, 1], [1, 1, 1, 0], [0, 1, 0, 0], [0, 1, 0, 0]],
                             [[1, 0, 0, 0], [1, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                             [[0, 0, 0, 1], [0, 0, 1, 0], [1, 1, 0, 0], [1, 0, 0, 0]],
                             [[0, 0, 0, 1], [1, 1, 1, 0], [1, 1, 0, 0], [1, 0, 0, 0]],
                             [[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [0, 0, 0, 1]],
                             [[0, 1, 0, 0], [0, 1, 1, 0], [0, 1, 1, 1], [1, 0, 0, 0]],
                             [[0, 0, 0, 1], [0, 1, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]]])
    if not mean_subtracted:
        for tri_image in two_by_two:
            unpaddedImageset.extend(getRotationsAndPad(tri_image))
        for tri_image in three_by_two:
            unpaddedImageset.extend(getRotationsAndPad(tri_image))
        for tri_image in two_by_four:
            unpaddedImageset.extend(getRotationsAndPad(tri_image))
        for tri_image in three_by_three:
            unpaddedImageset.extend(getRotationsAndPad(tri_image))
        for tri_image in three_by_four:
            unpaddedImageset.extend(getRotationsAndPad(tri_image))
        for tri_image in four_by_four:
            unpaddedImageset.extend(getRotationsAndPad(tri_image))
    else:
        for tri_image in two_by_two:
            unpaddedImageset.extend(getRotationsAfterMeanSubtractAndPad(tri_image))
        for tri_image in three_by_two:
            unpaddedImageset.extend(getRotationsAfterMeanSubtractAndPad(tri_image))
        for tri_image in two_by_four:
            unpaddedImageset.extend(getRotationsAfterMeanSubtractAndPad(tri_image))
        for tri_image in three_by_three:
            unpaddedImageset.extend(getRotationsAfterMeanSubtractAndPad(tri_image))
        for tri_image in three_by_four:
            unpaddedImageset.extend(getRotationsAfterMeanSubtractAndPad(tri_image))
        for tri_image in four_by_four:
            unpaddedImageset.extend(getRotationsAfterMeanSubtractAndPad(tri_image))
    imageSet = []
    for tri_image in unpaddedImageset:
        imageSet.append(np.pad(tri_image, (2, 2), constant_values=(0, 0)))
    if gridwide:
        for i in range(len(imageSet)):
            imageSet[i] = mean_subtract(imageSet[i])
    return np.asarray(imageSet)

def calculateTriangleG(imageSet: NDArray, imageProduct=lambda x, y: ncc(x, y)):
    G = []
    for image1 in imageSet:
        for image2 in imageSet:
            G.append(imageProduct(image1, image2))
    return np.reshape(G, (len(imageSet), len(imageSet)))

def getRotationsAndPad(tri_image: NDArray):
    ls = [padToFour(tri_image)]
    for i in range(0, 3):
        tri_image = np.rot90(tri_image)
        ls.append(padToFour(tri_image))
    return ls

def getRotationsAfterMeanSubtractAndPad(tri_image: NDArray):
    mean_subtracted = mean_subtract(padToFour(tri_image))
    ls = [mean_subtracted]
    for i in range(0, 3):
        mean_subtracted = np.rot90(mean_subtracted)
        ls.append(mean_subtracted)
    return ls

def padToFour(tri_image: NDArray):
    shape = tri_image.shape
    return np.pad(tri_image, ((4 - shape[0], 0), (0, 4 - shape[1])), constant_values=(0, 0))

def calculateTriangleAWithBF():
    G = calculateTriangleG(generateImageSet())
    G_prime = pencorr(G, 192)
    eigenvalues, eigenvectors = np.linalg.eigh(G_prime)
    eigenvalues[eigenvalues < 0] = 0
    D = np.diag(np.sqrt(eigenvalues))
    A = np.matmul(D, eigenvectors.transpose())
    return A

def calculateAfromG(G):
    G_prime = pencorr(G, 192)
    eigenvalues, eigenvectors = np.linalg.eigh(G_prime)
    eigenvalues[eigenvalues < 0] = 0
    D = np.diag(np.sqrt(eigenvalues))
    A = np.matmul(D, eigenvectors.transpose())
    return A

"""
Task: Compare triangles that you think are "similar" and see if they get a high NCC score from both
direct NCC calculation and through vector embedding.
"""

class TriangleImageSet:

    def __init__(self):
        self.fullSet = generateImageSet()
        self.two_by_two = np.array([[[1, 0],[1, 1]]])
        self.three_by_two = np.array([[[1, 0], [1, 0], [1, 1]], [[1, 0], [1, 1], [1, 0]], [[1, 1], [1, 0], [1, 0]],
                             [[1, 0], [1, 0], [0, 1]], [[0, 1], [1, 0], [1, 0]]])
        self.two_by_four = np.array([[[1, 1, 1, 1], [1, 0, 0, 0]], [[1, 1, 1, 1], [0, 1, 0, 0]], [[1, 1, 1, 1], [0, 0, 1, 0]],
                            [[1, 1, 1, 1], [0, 0, 0, 1]], [[0, 0, 1, 1], [1, 0, 0, 0]], [[1, 1, 0, 0], [0, 0, 0, 1]],
                            [[0, 1, 1, 1], [1, 0, 0, 0]], [[1, 1, 1, 0], [0, 0, 0, 1]]])
        self.three_by_three = np.array([[[1, 0, 0], [1, 1, 0], [1, 1, 1]], [[1, 0, 0], [1, 1, 1], [1, 0, 0]],
                               [[1, 0, 0], [1, 1, 0], [0, 0, 1]], [[0, 0, 1], [1, 1, 0], [0, 0, 1]],
                               [[0, 1, 0], [1, 1, 0], [0, 0, 1]]])
        self.three_by_four = np.array([[[1, 1, 1, 1], [1, 1, 0, 0], [1, 0, 0, 0]], [[1, 1, 1, 1], [0, 1, 1, 0], [0, 1, 0, 0]],
                              [[1, 1, 1, 1], [0, 1, 1, 0], [0, 0, 1, 0]], [[1, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 1]],
                              [[0, 0, 1, 0], [0, 1, 1, 1], [1, 0, 0, 0]], [[0, 0, 1, 0], [1, 1, 1, 0], [0, 0, 0, 1]],
                              [[1, 0, 0, 0], [1, 1, 1, 1], [1, 0, 0, 0]], [[0, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0]],
                              [[1, 0, 0, 0], [1, 1, 0, 0], [0, 0, 0, 1]], [[0, 0, 1, 1], [0, 1, 0, 0], [1, 0, 0, 0]],
                              [[0, 1, 0, 0], [1, 1, 1, 0], [0, 0, 0, 1]], [[0, 1, 0, 0], [0, 1, 1, 1], [1, 0, 0, 0]],
                              [[1, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], [[0, 1, 1, 1], [0, 1, 0, 0], [1, 0, 0, 0]],
                              [[1, 1, 1, 0], [0, 0, 1, 0], [0, 0, 0, 1]], [[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                              [[0, 0, 0, 1], [0, 1, 0, 0], [1, 0, 0, 0]]])
        self.four_by_four = np.array([[[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]],
                             [[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 1], [1, 0, 0, 0]],
                             [[1, 0, 0, 0], [1, 1, 1, 1], [1, 1, 0, 0], [1, 0, 0, 0]],
                             [[0, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                             [[0, 1, 0, 0], [0, 1, 0, 0], [1, 1, 1, 0], [0, 0, 0, 1]],
                             [[0, 0, 0, 1], [1, 1, 1, 0], [0, 1, 0, 0], [0, 1, 0, 0]],
                             [[1, 0, 0, 0], [1, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                             [[0, 0, 0, 1], [0, 0, 1, 0], [1, 1, 0, 0], [1, 0, 0, 0]],
                             [[0, 0, 0, 1], [1, 1, 1, 0], [1, 1, 0, 0], [1, 0, 0, 0]],
                             [[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [0, 0, 0, 1]],
                             [[0, 1, 0, 0], [0, 1, 1, 0], [0, 1, 1, 1], [1, 0, 0, 0]],
                             [[0, 0, 0, 1], [0, 1, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]]])
        self.G = calculateTriangleG(self.fullSet)
        self.Gprime = pencorr(self.G, 192)
        self.A = produceA(self.Gprime)

    def get2x2sample(self):
        imageSet = getRotationsAndPad(self.two_by_two[0])
        a = []
        for tri_image in imageSet:
            a.append(np.pad(tri_image, (2, 2), constant_values=(0, 0)))
        return a

    def get3x2sample(self):
        list = []
        sample = random.choice(self.three_by_two)
        list.append(getRotationsAndPad(sample))
        return list

    def get4x2sample(self):
        list = []
        sample = random.choice(self.two_by_four)
        list.append(getRotationsAndPad(sample))
        return list

    def get3x3sample(self):
        list = []
        sample = random.choice(self.three_by_three)
        list.append(getRotationsAndPad(sample))
        return list

    def get4x4sample(self):
        list = []
        sample = random.choice(self.four_by_four)
        list.append(getRotationsAndPad(sample))
        return list

    def get4x3sample(self):
        list = []
        sample = random.choice(self.three_by_four)
        list.append(getRotationsAndPad(sample))
        return list

    def getFullSet(self):
        return self.fullSet

    def getG(self):
        return self.G

    def getGprime(self):
        return self.Gprime

    def getA(self):
        return self.A

    def generate_new_G(self, imageProduct):
        return calculateTriangleG(self.fullSet, imageProduct=imageProduct)


"""
Sanity check 1: Are rotations of images similar to each other?
    Interpretation: Not really. NCC score of rotations can go as low as 0.3.
Sanity check 2: Are larger versions of the same triangle similar to smaller versions and vice versa?
    Interpretation: Relatively, but as size differs, score differs more.
Sanity check 3: More triangle comparisons to visually similar triangles
    Interpretation: These triangles show a 0.75 or more NCC score.
Sanity check 4: Flipped triangles
    Interpretation: In the first test, the ncc score is high. However, when tested on the smaller triangle (4x3 size),
    the NCC score is only 0.5 (for both). Smaller triangles are likely more sensitive to differences.
Sanity check 5: Smaller triangles
    Interpretation: Possibly confirms previous test. All matches have NCC score around 0.6.
"""

def triangleSanityTest1():
    imageSet = TriangleImageSet()
    sample = imageSet.get2x2sample()
    firstImage = sample[0]
    b = []
    for i in sample:
        b.append(ncc(i, firstImage))
    x = get_embedding_estimate(firstImage)
    for i in sample:
        b.append((np.dot(get_embedding_estimate(i), x)))
    return firstImage, b

def triangleSanityTest2():
    two = np.pad(padToFour(np.array([[1, 0],[1, 1]])), (2, 2), constant_values=(0, 0))
    three = np.pad(padToFour(np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1]])), (2, 2), constant_values=(0, 0))
    four = np.pad(padToFour(np.array([[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]])), (2, 2), constant_values=(0, 0))

    a = (ncc(two, two), ncc(two, three), ncc(two, four))
    b = (ncc(four, two), ncc(four, three), ncc(four, four))
    two2 = get_embedding_estimate(two)
    three2 = get_embedding_estimate(three)
    four2 = get_embedding_estimate(four)
    c = (np.dot(two2, two2), np.dot(two2, three2), np.dot(two2, four2))
    return a, b, c

def triangleSanityTest3():
    one = np.array([[1, 0], [1, 0], [1, 1]])
    two = np.rot90(np.array([[1, 1, 1, 1], [1, 1, 0, 0], [1, 0, 0, 0]]))
    three = np.rot90(np.array([[1, 1], [1, 0], [1, 0]]))
    four = np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1]])

    imageSet = [one, two, three, four]
    imgSet = []
    for tri_image in imageSet:
        imgSet.append(np.pad(padToFour(tri_image), (2, 2), constant_values=(0, 0)))
    one2 = get_embedding_estimate(imgSet[0])
    est = []
    for tri_image in imgSet:
        est.append(np.dot(one2, get_embedding_estimate(tri_image)))
    return (ncc(imgSet[0], imgSet[0]), ncc(imgSet[0], imgSet[1]), ncc(imgSet[0], imgSet[2]),
            ncc(imgSet[0], imgSet[3]), est), imgSet

def triangleSanityTest4():
    one = np.array([[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 1], [1, 0, 0, 0]])
    two = np.array([[1, 0, 0, 0], [1, 1, 1, 1], [1, 1, 0, 0], [1, 0, 0, 0]])
    three = np.rot90(np.array([[0, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0]]))
    four = np.rot90(np.array([[0, 0, 1, 1], [0, 1, 0, 0], [1, 0, 0, 0]]))
    five = np.array([[1, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    imageSet = [one, two, three, four, five]
    imgSet = []
    for tri_image in imageSet:
        imgSet.append(np.pad(padToFour(tri_image), (2, 2), constant_values=(0, 0)))
    one2 = get_embedding_estimate(imgSet[0])
    three2 = get_embedding_estimate(imgSet[2])
    est = (np.dot(one2, get_embedding_estimate(imgSet[1])), np.dot(three2, get_embedding_estimate(imgSet[3])),
           np.dot(three2, get_embedding_estimate(imgSet[4])))
    return (ncc(imgSet[0], imgSet[0]), ncc(imgSet[0], imgSet[1]), ncc(imgSet[2], imgSet[2]),
            ncc(imgSet[2], imgSet[3]), ncc(imgSet[2], imgSet[4]), est), imgSet

def triangleSanityTest5():
    one = np.array([[1, 0], [1, 0], [0, 1]])
    two = np.array([[0, 1], [1, 0], [1, 0]])
    three = np.array([[1, 0, 0], [1, 1, 0], [0, 0, 1]])
    four = np.rot90(np.array([[0, 0, 1, 1], [1, 0, 0, 0]]))
    five = np.rot90(two, k=-1)

    imageSet = [one, two, three, four, five]
    imgSet = []
    for tri_image in imageSet:
        imgSet.append(np.pad(padToFour(tri_image), (2, 2), constant_values=(0, 0)))
    est = []
    one2 = get_embedding_estimate(imgSet[0])
    for tri_image in imgSet:
        est.append(np.dot(one2, get_embedding_estimate(tri_image)))
    return (ncc(imgSet[0], imgSet[0]), ncc(imgSet[0], imgSet[1]), ncc(imgSet[0], imgSet[2]),
            ncc(imgSet[0], imgSet[3]), ncc(imgSet[0], imgSet[4]), est), imgSet

"""
Mean subtracted tests: Mean subtracted triangles give a generally quite similar Relative Positioning Score as non
mean subtracted triangles. At k=3, normal triangles win out, but at k=5, mean subtracted triangles give a slightly
higher score.
Next, try mean subtracting across the entire grid.

When mean subtracting across the whole grid, Relative Positioning score is strictly worse for both k=3 and k=5
"""

# This estimate is using the Lagrangian method and is the same as the one in SampleEstimator.py
def get_embedding_estimate(image):
    triangleImageSet = TriangleImageSet()
    trainingImageSet = triangleImageSet.getFullSet()
    imageProductVector = ip.calculate_image_product_vector(image, trainingImageSet, ncc)
    estimateVector = Lagrangian_Method2(triangleImageSet.getA(), imageProductVector)[0]
    return estimateVector

# This NCC calculation is the same as the one in ImageProducts.py
def ncc(mainImg: NDArray, tempImg: NDArray):

    if np.count_nonzero(mainImg) == 0:
        if np.count_nonzero(tempImg) == 0:
            return 1
        return 0

    mainImg = np.pad(mainImg, max(len(mainImg), len(mainImg[0])), 'wrap')

    mainImg = np.asarray(mainImg, np.single)
    tempImg = np.asarray(tempImg, np.single)

    corr = cv2.matchTemplate(mainImg, tempImg, cv2.TM_CCORR_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(corr)

    return max_val

def mean_subtract(matrix: NDArray):
    mean = np.mean(matrix)
    return matrix - np.ones(matrix.shape) * mean

def generate_diagnostics(image_type, image_product, k=5, embedding=None):
    """
    :param image_type: Image Set to investigate
    :param image_product: Image Product to investigate
    :return: Returns several metric which we may want to observe. To be updated over time.
    :returns G: The Image Product Matrix
    :returns G_prime: The Nearest Correlation Matrix found using Pencorr
    :returns A: The embedding matrix found by decomposition
    :returns x: An array of tuples containing the minimum and maximum values found in corresponding rows of A
    :returns r: The range of values from x
    :returns s: The sum of all ranges. Rough estimator of area of hypersphere occupied by embeddings
    :returns prod: The product of all elements of r. Another rough estimator of proportion of Hypersphere occupied
    :returns nonzero: Number of non-zero elements of r. Indicates the number of dimensions the embeddings occupy
    :returns eigenvalues/eigenvectors: Eigenvalues and eigenvectors of G_prime
    """
    image_set = ig.get_image_set(image_type)
    G = utils.generate_image_product_matrix(image_set, image_product, fputils.get_image_product_filepath(image_type, [], image_product))
    if embedding is None:
        embedding = "pencorr_" + str(len(image_set))
    A = utils.generate_embedding_matrix(G, embedding, fputils.get_embedding_matrix_filepath(image_type, [], image_product, embedding))
    G_prime = np.matmul(np.atleast_2d(A).T, np.atleast_2d(A))
    x = np.array([(np.min(b), np.max(b)) for b in A])   # Range of values which the embeddings take in each dimension
    r = np.array([np.max(b) - np.min(b) for b in A])    # Magnitude of this range
    s = sum([np.max(b) - np.min(b) for b in A])         # Sum of all ranges, to gauge how much of the Hypersphere we are using
    nonzero = np.count_nonzero(r)                       # Number of dimensions used
    prod = np.sum([math.log10(i) for i in r[r != 0]])
    prod2 = prod - math.log10((math.pi ** (nonzero / 2)) / math.gamma(1 + nonzero / 2))
    eigenvalues, eigenvectors = np.linalg.eigh(G)
    k_score = metrics.get_mean_normed_k_neighbour_score(G, G_prime, k)
    return G, G_prime, A, x, r, s, prod, prod2, nonzero, eigenvalues, eigenvectors, k_score, embedding

def compare_image_products(image_set, image_product_list, embeddings=None):
    if embeddings is None:
        embeddings = [("pencorr_" + str(len(image_set)))]
    data = {
        "image_product": [],
        "embedding": [],
        "sum": [],
        "non_zero": [],
        "non_negative": [],
        "prod": [],
        "k_scores": []
    }
    for image_product in image_product_list:
        img_prod = ip.get_image_product(image_product)
        for embedding in embeddings:
            G, G_prime, A, x, r, s, prod, prod2, nonzero, eigenvalues, eigenvectors, k_score, emb = (
                generate_diagnostics(image_set, img_prod, embedding=embedding))
            data["image_product"].append(image_product)
            data["embedding"].append(emb)
            data["sum"].append(s)
            data["non_zero"].append(nonzero)
            data["non_negative"].append(np.sum([eigenvalues >= 0]))
            data["prod"].append(prod)
            data["k_scores"].append(k_score)
    df = pd.DataFrame(data)
    return df

def plot_k_on_values(k: int, image_type: str, image_product_list: list, plot=None, embeddings=None):
    if embeddings is None:
        embeddings = [("pencorr_" + str(len(ig.get_image_set(image_type))))]
    data = {
        "image_product": [],
        "embedding": [],
        "sum": [],
        "non_zero": [],
        "non_negative": [],
        "prod": [],
        "prod2": [],
        "k_scores": []
    }
    for image_product in image_product_list:

        for embedding in embeddings:
            G, G_prime, A, x, r, s, prod, prod2, nonzero, eigenvalues, eigenvectors, k_score, red = (
                generate_diagnostics(image_type, image_product, k=k, embedding=embedding))
            data["image_product"].append(image_product)
            data["embedding"].append(embedding)
            data["sum"].append(s)
            data["non_zero"].append(nonzero)
            data["non_negative"].append(np.sum([eigenvalues >= 0]))
            data["prod"].append(prod)
            data["prod2"].append(prod2)
            data["k_scores"].append(k_score)
    if plot is None:
        plot = "non_zero"
    plt.plot(data[plot], data["k_scores"], "r+")
    plt.plot(np.unique(data[plot]), np.poly1d(np.polyfit(data[plot], data["k_scores"], 1))(np.unique(data[plot])))
    idealPlot = [1 for i in range(len(data[plot]))]  # for plotting the max possible score
    plt.plot(data[plot], idealPlot, color='b', linestyle=':', label="Ideal")
    plt.show()
    df = pd.DataFrame(data)
    return df


"""
Some useful links and ideas
https://developers.google.com/machine-learning/clustering/similarity/measuring-similarity
https://learn.microsoft.com/en-us/azure/search/vector-search-overview#approximate-nearest-neighbors

Vector similarity can help to determine close images or similar images. Cosine similarity is equal to NCC score.
Consider any image, then calculate the closest image?

For similarity metric, consider developing relative positioning score with more specific metric calculations.

Properties of embeddings:
Note that xtx = 1. So, arranging them into a diagonal matrix, we get Dx squared = ??
"""
import cv2
import random
import numpy as np
from numpy.typing import NDArray
from sklearn.preprocessing import normalize
from src.data_processing.EmbeddingFunctions import pencorr

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
def generateImageSet():
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
    imageSet = []
    for tri_image in unpaddedImageset:
        imageSet.append(np.pad(tri_image, (2, 2), constant_values=(0, 0)))
    return np.asarray(imageSet)

def calculateTriangleG(imageSet: NDArray):
    G = []
    for image1 in imageSet:
        for image2 in imageSet:
            G.append(ncc(image1, image2))
    return np.reshape(G, (len(imageSet), len(imageSet)))

def getRotationsAndPad(tri_image: NDArray):
    ls = [padToFour(tri_image)]
    for i in range(0, 3):
        tri_image = np.rot90(tri_image)
        ls.append(padToFour(tri_image))
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

"""
Sanity check 1: Are rotations of images similar to each other?
    Interpretation: Not really. NCC score of rotations can go as low as 0.3.
Sanity check 2: Are larger versions of the same triangle similar to smaller versions and vice versa?
    Interpretation: No, although the two triangles are similar visually, the NCC scores are not particularly high.
Sanity check 3: More triangle comparisons to visually similar triangles
"""

def triangleSanityTest1():
    imageSet = TriangleImageSet()
    sample = imageSet.get2x2sample()
    firstImage = sample[0]
    b = []
    for i in sample:
        b.append(ncc(i, firstImage))
    return firstImage, b

def triangleSanityTest2():
    two = np.pad(padToFour(np.array([[1, 0],[1, 1]])), (2, 2), constant_values=(0, 0))
    three = np.pad(padToFour(np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1]])), (2, 2), constant_values=(0, 0))
    four = np.pad(padToFour(np.array([[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]])), (2, 2), constant_values=(0, 0))

    a = (ncc(two, two), ncc(two, three), ncc(two, four))
    b = (ncc(four, two), ncc(four, three), ncc(four, four))
    return a, b

def triangleSanityTest3():
    one = np.array([[1, 0], [1, 0], [1, 1]])
    two = np.rot90(np.array([[1, 1, 1, 1], [1, 1, 0, 0], [1, 0, 0, 0]]))
    three = np.rot90(np.array([[1, 1], [1, 0], [1, 0]]))
    four = np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1]])

    imageSet = [one, two, three, four]
    imgSet = []
    for tri_image in imageSet:
        imgSet.append(np.pad(padToFour(tri_image), (2, 2), constant_values=(0, 0)))
    return (ncc(imgSet[0], imgSet[0]), ncc(imgSet[0], imgSet[1]), ncc(imgSet[0], imgSet[2]),
            ncc(imgSet[0], imgSet[2])), imgSet

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
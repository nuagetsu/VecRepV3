import os
from typing import List

from data_processing import FilepathUtils
from src.data_processing.EmbeddingFunctions import get_eig_for_symmetric
import numpy as np
from numpy.typing import NDArray


def get_k_neighbour_score(imageProducts: NDArray, embeddingDotProducts: NDArray, k: int) -> float:
    """
    :param imageProducts: 1 by N array of image product scores between an image and all another images
    :param embeddingDotProducts: 1 by N array of dot products with an embedding and all other embeddings
    :return: The k neighbour score, as defined in the readme
    Get the index of the top K elements in the embeddingsDotProducts (DP) array
    Get the index of the top K + x elements in the imageProducts (IP) array, where x is the number of elements in the IP
    array with the same value as the Kth largest element in the IP array
    Find the intersection between the two above arrays
    """
    k = k + 1  # This takes into account that the closest neighbour to the vector is itself

    # Get the index of the k largest elements in each list
    imgProd_max_index = np.argpartition(imageProducts, -k)[-k:]
    embProd_max_index = np.argpartition(embeddingDotProducts, -k)[-k:]
    # Get the kth largest element of the image products array
    kth_element = imageProducts[imgProd_max_index[0]]
    # Get the index of elements with the same value as the kth element
    kth_element_index = np.where(imageProducts == kth_element)
    # Add the kth elements to the set of k closest neighbours for the image products array
    imgProd_max_index = np.union1d(imgProd_max_index, kth_element_index)
    # Get number of neighbours which remain closest
    similar_neighbours = np.intersect1d(imgProd_max_index, embProd_max_index)

    res = len(similar_neighbours) - 1  # Take into account that the closest neighbour is itself
    return res


def get_normed_k_neighbour_score(imageProducts: NDArray, embeddingDotProducts: NDArray, k: int) -> float:
    """
    :param imageProducts: 1 by N array of image product scores between an image and all another images
    :param embeddingDotProducts: 1 by N array of dot products with an embedding and all other embeddings
    :return: The k neighbour score, as defined in the readme
    Same process as k neighbour score, except the final result is between 0 and 1
    """
    kNeighScore = get_k_neighbour_score(imageProducts, embeddingDotProducts, k)

    res = float(kNeighScore / k)
    return res
def get_frob_distance(imageProductMatrix: NDArray, embeddingMatrix: NDArray) -> float:
    """
    :param imageProductMatrix: Image product array to be compared
    :param embeddingMatrix: Embedding matrix to be compared
    :return: The frobenius distance between the two vectors
    """
    diff = imageProductMatrix - embeddingMatrix
    frobNorm = np.linalg.norm(diff)
    return frobNorm


def apply_k_neighbour(imageProductArray: NDArray, embeddingDotProductArray: NDArray, startingK: int,
                      endingK: int) -> List:
    """
    :param endingK: Ending K neighbour score, inclusive
    :param startingK: Starting K neighbour score, inclusive. finds the neighbour score then increments by one until endingK
    :param imageProductArray: Image product array to be compared
    :param embeddingDotProductArray: A^tA, where A is the embedding matrix
    :return: A list which has the k neighbour score for each image, for each value of k from startingK to endingK.
    In the form of: [{startingK, [list of k neigbour scores]}, {startingK + 1, [list of k + 1 neigbour scores]} ... ,
    {[endingK, [list of endingK neigbour scores]}]
    """
    if startingK >= endingK:
        raise ValueError("Starting K should be lower than ending K")
    if endingK + 1 > len(imageProductArray):
        raise ValueError("Ending K + 1 should be less than the number of images")
    output = []
    kVals = range(startingK, endingK + 1)
    for kval in kVals:
        scores = []
        for imageNumber in range(len(imageProductArray)):
            scores.append(
                get_k_neighbour_score(imageProductArray[imageNumber], embeddingDotProductArray[imageNumber], kval))
        output.append({"kval": kval, "neighbourScore": scores})
    return output


class PlottingData:
    def __init__(self, *, initialEigenvalues, finalEigenvalues, frobDistance, kNeighbourScores, numImages, imagesFilepath):
        self.initialEigenvalues = np.array(initialEigenvalues)
        self.finalEigenvalues = np.array(finalEigenvalues)
        self.frobDistance = frobDistance
        self.aveFrobDistance = frobDistance / (numImages ** 2)
        self.kNeighbourScores = kNeighbourScores
        self.imagesFilepath = imagesFilepath


def get_plotting_data(*, imageType: str, filters=None, imageProductType: str, embeddingType: str):
    """
    :param imageType: imageType to of the images
    :param filters: filters applied to the images
    :param imageProductType: the image product type which we aim to estimate
    :param embeddingType: The method used to generate the vector embeddings
    :return: A plotting data object which can be used for graphs to evaluate if the embeddings are a good estimate.

    Note that the data must be saved beforehand in the data directory to obtain the plotting data. Use VecRep.py to generate the data first
    """
    imageProductMatrix, embeddingMatrix = get_ipm_and_embeddings(imageType=imageType, filters=filters,
                                                                 imageProductType=imageProductType,
                                                                 embeddingType=embeddingType)

    initialEigenvalues, eigVec = get_eig_for_symmetric(imageProductMatrix)
    dotProdMatrix = np.matmul(embeddingMatrix.T, embeddingMatrix)
    finalEigenvalues, eigVec = get_eig_for_symmetric(dotProdMatrix)
    frobDistance = get_frob_distance(imageProductMatrix, dotProdMatrix)
    numImages = len(imageProductMatrix[0])

    # Sweep from k=1 to k = numimages/5 by default. If num images is small then sweep from 1 - 2
    kNeighbourScores = apply_k_neighbour(imageProductMatrix, dotProdMatrix, 1, max(int(numImages / 5), 2))

    imagesFilepath = FilepathUtils.get_image_set_filepath(imageType=imageType, filters=filters)

    output = PlottingData(initialEigenvalues=initialEigenvalues, finalEigenvalues=finalEigenvalues,
                          frobDistance=frobDistance, kNeighbourScores=kNeighbourScores, numImages=numImages,
                          imagesFilepath=imagesFilepath)
    return output


def get_ipm_and_embeddings(*, imageType: str, filters=None, imageProductType: str, embeddingType: str):
    embeddingFilepath = FilepathUtils.get_embedding_matrix_filepath(imageType=imageType, filters=filters,
                                                                    imageProductType=imageProductType,
                                                                    embeddingType=embeddingType)
    imageProductFilepath = FilepathUtils.get_image_product_filepath(imageType=imageType, filters=filters,
                                                                    imageProductType=imageProductType)

    if os.path.isfile(imageProductFilepath):
        imageProductMatrix = np.loadtxt(imageProductFilepath)
    else:
        raise ValueError(imageProductFilepath + " does not exist. Generate data first before graphing")

    if os.path.isfile(embeddingFilepath):
        embeddingMatrix = np.loadtxt(embeddingFilepath)
    else:
        raise ValueError(embeddingFilepath + " does not exist. Generate data first before graphing")

    return imageProductMatrix, embeddingMatrix

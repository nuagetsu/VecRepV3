from statistics import mean

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
    if k > len(imageProducts) + 1:
        raise ValueError("Value of k in K neighbour score must be less than the number of images")
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

    res = max(len(similar_neighbours) - 1, 0)  # Take into account that the closest neighbour is itself
    return res


def get_normed_k_neighbour_score(imageProducts: NDArray, embeddingDotProducts: NDArray, k: int) -> float:
    """
    :param k: Value of k in k neighbour score
    :param imageProducts: 1 by N array of image product scores between an image and all another images
    :param embeddingDotProducts: 1 by N array of dot products with an embedding and all other embeddings
    :return: The k neighbour score, as defined in the readme
    Same process as k neighbour score, except the final result is between 0 and 1
    """
    kNeighScore = get_k_neighbour_score(imageProducts, embeddingDotProducts, k)

    res = float(kNeighScore / k)
    return res

def get_mean_normed_k_neighbour_score(matrixG: NDArray, matrixGprime:NDArray, k: int) -> float:
    kNeighArray = []
    for rowIndex in range(len(matrixG)):
        kNeighArray.append(get_normed_k_neighbour_score(matrixG[rowIndex], matrixGprime[rowIndex], k))
    return mean(kNeighArray)

def get_frob_distance(imageProductMatrix: NDArray, embeddingMatrix: NDArray) -> float:
    """
    :param imageProductMatrix: Image product array to be compared
    :param embeddingMatrix: Embedding matrix to be compared
    :return: The frobenius distance between the two vectors
    """
    diff = imageProductMatrix - embeddingMatrix
    frobNorm = np.linalg.norm(diff)
    return frobNorm

def get_progressive_range(startingConstr: int, endingConstr: int, interval: int):
    """
    :param startingConstr: Starting constraint for progressive range
    :param endingConstr: Ending constraint for progressive range
    :param interval: Starting interval for progressive range that increases by 1 for each cycle of 10
    :return: The progressive range as a list
    """
    constraints = []
    top = startingConstr
    while top < endingConstr:
        curr = list(range(top, top + (interval * 10), interval))
        constraints.extend(curr)
        interval += 1
        top = curr[-1] + interval
    while constraints[-1] >= endingConstr:
        constraints.pop()
    return constraints

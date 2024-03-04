import numpy as np
from numpy.typing import NDArray

def k_neighbour_score(imageProducts: NDArray, embeddingDotProducts: NDArray, k:int) -> float:
    """
    :param imageProducts: 1 by N array of image product scores between an image and all another images
    :param embeddingDotProducts: 1 by N array of dot products with an embedding and all other embeddings
    :return: The k neighbour score, as defined in the readme
    Get the index of the top K elements in the embeddingsDotProducts (DP) array
    Get the index of the top K + x elements in the imageProducts (IP) array, where x is the number of elements in the IP
    array with the same value as the Kth largest element in the IP array
    Find the intersection between the two above arrays
    Divide the size of the intersection by K
    """
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

    return len(similar_neighbours) / k
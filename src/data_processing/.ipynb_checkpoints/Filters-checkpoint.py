import re

import numpy as np
from numpy.typing import NDArray

from src.helpers.NumIslands import countIslands


def get_filtered_image_sets(*, imageSet: NDArray, filters: list[str]) -> NDArray:
    if filters is not None:
        for filter in filters:
            imageSet = apply_filter(imageSet, filter)
    return imageSet


def apply_filter(imageSet: NDArray, filter: str) -> NDArray:
    """
    :param imageSet: image set to be filtered
    :param filter: string which corresponds to a certain filter
    :return: the image set with the respective filter function applied

    Takes in a string and looks up the correct filter function to apply
    """
    if filter == 'unique':
        imageSet = apply_translationally_unique_filter(imageSet)
    elif re.search('[0-9]?[0-9]max_ones$', filter) is not None:
        onesMaxPercentage = int(re.search(r'\d+', filter).group())
        imageSet = apply_max_ones_filter(imageSet, onesMaxPercentage)
    elif filter == 'one_island':
        imageSet = apply_one_island_filter(imageSet)
    else:
        raise ValueError(filter + " is not a valid filter type")
    return imageSet


def apply_one_island_filter(imageSet: NDArray) -> NDArray:
    """
    Applies the "one island" filter. Filters out "noisy" images that are not just 1 object.
    :param imageSet: Image set to filter
    :return: Filtered image set
    """
    final_arr = []
    for image in imageSet:
        if countIslands(image) == 1:
            final_arr.append(image)
    final_arr = np.array(final_arr)
    return final_arr


def canonical_translation(image):
    """
    Computes the canonical (minimal) byte representation over all translations
    of a square image.
    
    :param image: Square NumPy array.
    :return: The minimal bytes representation (in lexicographical order) among all translations.
    """
    n = image.shape[0]
    best = None
    # loop row shifts
    for dr in range(n):
        rolled_row = np.roll(image, shift=dr, axis=0)
        # then column
        for dc in range(n):
            variant = np.roll(rolled_row, shift=dc, axis=1)
            variant_bytes = variant.tobytes()
            if best is None or variant_bytes < best:
                best = variant_bytes
    return best

def apply_translationally_unique_filter(imageSet: 'NDArray') -> 'NDArray':
    """
    Filters an image set to retain only translationally unique images by using a
    canonical representation computed for each image.
    
    :param imageSet: Array of square images.
    :return: Filtered image set with translational duplicates removed.
    """
    unique_images = []
    seen = set()

    for matrix in imageSet:
        canon = canonical_translation(matrix)
        if canon in seen:
            continue
        seen.add(canon)
        unique_images.append(matrix)
    
    return np.array(unique_images)

def apply_max_ones_filter(imageSet: NDArray, onesMaxPercentage: float) -> NDArray:
    """
    Applies the "max ones" filter which filters out all images with greater 1s than percentage allowed
    :param imageSet: Image set to be filtered
    :param onesMaxPercentage: Maximum percentage of 1s allowed
    :return: Filtered image set
    """
    total = len(imageSet[0]) * len(imageSet[0][0])
    total = (total * onesMaxPercentage) // 100
    finalArr = []
    for square in imageSet:
        if np.sum(square) <= total:
            finalArr.append(square)
    finalArr = np.array(finalArr)
    return finalArr

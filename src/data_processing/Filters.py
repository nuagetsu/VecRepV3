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
    final_arr = []
    for image in imageSet:
        if countIslands(image) == 1:
            final_arr.append(image)
    final_arr = np.array(final_arr)
    return final_arr


def apply_translationally_unique_filter(imageSet: NDArray) -> NDArray:
    """
    For each item in the image set, if it is not in the all_permutations set, add it to the final list
    Then store all possible translations of the image in the all_permutations.
    Otherwise
    """
    unique = []
    squareLength = len(imageSet[0])
    all_permutations = set()

    for matrix in imageSet:
        original_matrix = np.copy(matrix)
        original_matrix = np.reshape(original_matrix, (1, squareLength ** 2))

        if tuple(original_matrix[0]) in all_permutations:
            continue

        else:
            unique.append(matrix)
            # All translational invariant permutations for given nxn matrix
            for dr in range(squareLength):
                matrix = np.roll(matrix, 1, axis=0)  # shift 1 place in vertical axis
                for dc in range(squareLength):
                    matrix = np.roll(matrix, 1, axis=1)  # shift 1 place in horizontal axis
                    to_store = np.reshape(matrix, (1, squareLength ** 2))
                    all_permutations.add(tuple(to_store[0]))  # store in dictionary
    unique = np.array(unique)
    return unique


def apply_max_ones_filter(imageSet: NDArray, onesMaxPercentage: float) -> NDArray:
    total = len(imageSet[0]) * len(imageSet[0][0])
    total = (total * onesMaxPercentage) // 100
    finalArr = []
    for square in imageSet:
        if np.sum(square) <= total:
            finalArr.append(square)
    finalArr = np.array(finalArr)
    return finalArr

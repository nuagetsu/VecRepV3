import itertools
import re

import numpy as np
from numpy.typing import NDArray

from src.helpers.IslandCreator import grid_creation


def get_binary_image_set(imageLength: int, maxOnesPercentage=100) -> NDArray[int]:
    if maxOnesPercentage > 100:
        raise ValueError(str(maxOnesPercentage) + " > 100. Maximum percentage of ones has to be less than 100")
    cutoff = maxOnesPercentage * (imageLength ** 2) // 100
    fullList = []
    for item in itertools.product([0, 1], repeat=imageLength ** 2):
        if np.sum(item) <= cutoff:
            fullList.append(np.reshape(item, (imageLength, imageLength)))
    fullList = np.array(fullList)
    return fullList


def get_image_set(imageType: str):
    if re.search('[0-9]?[0-9]bin[0-9]?[0-9]max_ones$', imageType) is not None:  # Searching if image type follows the
        # format of 3bin40max_ones
        imageLength = int(re.search(r'^\d+', imageType).group())
        maxOnesPercentage = int(re.search(r'\d+', imageType[2:]).group())
        image_set = get_binary_image_set(imageLength, maxOnesPercentage=maxOnesPercentage)
    elif re.search('[0-9]?[0-9]bin$', imageType) is not None:  # Searching if the image type follows the format 2bin
        imageLength = int(re.search(r'\d+', imageType).group())
        image_set = get_binary_image_set(imageLength)
    elif imageType == "triangle":
        image_set = get_triangle_image_set()
    elif imageType == "triangle_mean_subtracted":
        image_set = get_triangle_image_set(mean_subtracted=True)
    elif imageType == "triangle_gms":
        image_set = get_triangle_image_set(gridwide=True)
    else:
        raise ValueError(imageType + " is not a valid image type")
    return image_set

def get_island_image_set(imageType, numImages):
    """
    :param imageLength: side length of image grid
    :param percentOnes: Percent of ones
    :param numImages: number of images to generate
    :return: An image set of randomly generated islands with no repeats
    """
    if re.search('[0-9]?[0-9]island[0-9]?[0-9]max_ones$', imageType) is not None:  # Searching if image type follows the
        # format of 3bin40max_ones
        imageLength = int(re.search(r'^\d+', imageType).group())
        maxOnesPercentage = int(re.search(r'\d+', imageType[2:]).group())
        return np.array(grid_creation(imageLength, numImages, int(maxOnesPercentage / 100 * (imageLength ** 2))))
    else:
        raise ValueError("invalid image type")

def get_triangle_image_set(mean_subtracted=False, gridwide=False):
    """
    :return: The image set of 4x4 triangles within an 8x8 matrix.
    """
    unpaddedImageset = []
    two_by_two = np.array([[[1, 0], [1, 1]]])
    three_by_two = np.array([[[1, 0], [1, 0], [1, 1]], [[1, 0], [1, 1], [1, 0]], [[1, 1], [1, 0], [1, 0]],
                             [[1, 0], [1, 0], [0, 1]], [[0, 1], [1, 0], [1, 0]]])
    two_by_four = np.array([[[1, 1, 1, 1], [1, 0, 0, 0]], [[1, 1, 1, 1], [0, 1, 0, 0]], [[1, 1, 1, 1], [0, 0, 1, 0]],
                            [[1, 1, 1, 1], [0, 0, 0, 1]], [[0, 0, 1, 1], [1, 0, 0, 0]], [[1, 1, 0, 0], [0, 0, 0, 1]],
                            [[0, 1, 1, 1], [1, 0, 0, 0]], [[1, 1, 1, 0], [0, 0, 0, 1]]])
    three_by_three = np.array([[[1, 0, 0], [1, 1, 0], [1, 1, 1]], [[1, 0, 0], [1, 1, 1], [1, 0, 0]],
                               [[1, 0, 0], [1, 1, 0], [0, 0, 1]], [[0, 0, 1], [1, 1, 0], [1, 0, 0]],
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
            unpaddedImageset.extend(get_rotations_and_pad(tri_image))
        for tri_image in three_by_two:
            unpaddedImageset.extend(get_rotations_and_pad(tri_image))
        for tri_image in two_by_four:
            unpaddedImageset.extend(get_rotations_and_pad(tri_image))
        for tri_image in three_by_three:
            unpaddedImageset.extend(get_rotations_and_pad(tri_image))
        for tri_image in three_by_four:
            unpaddedImageset.extend(get_rotations_and_pad(tri_image))
        for tri_image in four_by_four:
            unpaddedImageset.extend(get_rotations_and_pad(tri_image))
    else:
        for tri_image in two_by_two:
            unpaddedImageset.extend(get_rotations_of_mean_subtracted_and_padded(tri_image))
        for tri_image in three_by_two:
            unpaddedImageset.extend(get_rotations_of_mean_subtracted_and_padded(tri_image))
        for tri_image in two_by_four:
            unpaddedImageset.extend(get_rotations_of_mean_subtracted_and_padded(tri_image))
        for tri_image in three_by_three:
            unpaddedImageset.extend(get_rotations_of_mean_subtracted_and_padded(tri_image))
        for tri_image in three_by_four:
            unpaddedImageset.extend(get_rotations_of_mean_subtracted_and_padded(tri_image))
        for tri_image in four_by_four:
            unpaddedImageset.extend(get_rotations_of_mean_subtracted_and_padded(tri_image))
    imageSet = []
    for tri_image in unpaddedImageset:
        imageSet.append(np.pad(tri_image, (2, 2), constant_values=(0, 0)))
    if gridwide:
        for i in range(len(imageSet)):
            imageSet[i] = imageSet[i] - np.ones(imageSet[i].shape) * np.mean(imageSet[i])
    return np.asarray(imageSet)

def get_rotations_and_pad(tri_image: NDArray):
    ls = [pad_to_four(tri_image)]
    for i in range(0, 3):
        tri_image = np.rot90(tri_image)
        ls.append(pad_to_four(tri_image))
    return ls

def pad_to_four(tri_image: NDArray):
    shape = tri_image.shape
    return np.pad(tri_image, ((4 - shape[0], 0), (0, 4 - shape[1])), constant_values=(0, 0))

def get_rotations_of_mean_subtracted_and_padded(tri_image: NDArray):
    tri_image = pad_to_four(tri_image)
    mean_subtracted = tri_image - np.ones(tri_image.shape) * np.mean(tri_image)
    ls = [mean_subtracted]
    for i in range(0, 3):
        mean_subtracted = np.rot90(mean_subtracted)
        ls.append(mean_subtracted)
    return ls

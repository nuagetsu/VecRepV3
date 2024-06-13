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


def get_quadrilaterals_image_set():
    """
    :return: The image set of all 4x4 quadrilaterals and triangles.
    """
    combinations = []
    indexes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    for comb in itertools.combinations(indexes, 4):
        combinations.append(comb)
    combinations = np.asarray(combinations)
    combinations = np.delete(combinations, np.where(is_straight_line(combinations))[0])
    image_set = []

    for combination in combinations:
        matrix = np.zeros((4, 4), dtype=int)
        for i in combination:
            x = i / 4
            y = i % 4
            matrix[x][y] = 1
        matrix = fill_shape(matrix)
        image_set.append(matrix)

    return np.asarray(image_set)

def is_straight_line(combination: NDArray):
    combination = combination.tolist()
    vertical_lines = [[0, 4, 8, 12], [1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15]]
    horizontal_lines = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]
    diagonal_lines = [[0, 5, 10, 15], [3, 6, 9, 12]]
    if combination in horizontal_lines or combination in diagonal_lines or combination in vertical_lines:
        return True
    return True

def fill_shape(matrix: NDArray):
    change = True
    while change:
        before = np.copy(matrix)
        for x in range(4):
            for y in range(4):
                if matrix[x][y]:
                    continue
                x_m1 = matrix[max(0, x - 1)][y]
                x_m2 = matrix[max(0, x - 2)][y]
                x_p1 = matrix[min(x + 1, 3)][y]
                x_p2 = matrix[min(x + 2, 3)][y]
                y_m1 = matrix[x][max(0, y - 1)]
                y_m2 = matrix[x][max(0, y - 2)]
                y_p1 = matrix[x][min(y + 1, 3)]
                y_p2 = matrix[x][min(y + 2, 3)]
                x_d1 = x - 1
                x_d2 = x - 2
                y_d1 = y - 1
                y_d2 = y - 2
                x_d3 = x + 1
                y_d3 = y + 1
                x_d4 = x + 2
                y_d4 = y + 2
                if x_d1 < 0 or y_d1 < 0:
                    x_d1 = x
                    y_d1 = y
                if x_d2 < 0 or y_d2 < 0:
                    x_d2 = x_d1
                    y_d2 = y_d1
                if x_d3 > 3 or y_d3 > 3:
                    x_d3 = x
                    y_d3 = y
                if x_d4 > 3 or y_d4 > 3:
                    x_d4 = x_d3
                    y_d4 = y_d3
                # Horizontal check
                left = y_m1 or y_m2
                right = y_p1 or y_p2
                # Vertical check
                up = x_m1 or x_m2
                down = x_p1 or x_p2
                # Diagonal check
                diag1 = matrix[x_d1][y_d1]
                diag2 = matrix[x_d2][y_d2]
                diag3 = matrix[x_d3][y_d3]
                diag4 = matrix[x_d4][y_d4]
                diag5 = matrix[x_d3][y_d1]
                diag6 = matrix[x_d4][y_d2]
                diag7 = matrix[x_d1][y_d3]
                diag8 = matrix[x_d2][y_d4]
                upleft = diag1 or diag2
                downright = diag3 or diag4
                downleft = diag5 or diag6
                upright = diag7 or diag8
                if (left and right) or (up and down):
                    matrix[x][y] = 1
                elif (upleft and downright) or (downleft and upright):
                    matrix[x][y] = 1
        if np.array_equal(before, matrix):
            change = False
    return matrix





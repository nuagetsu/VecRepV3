import itertools
import logging
import random
import re

import numpy as np
from numpy.typing import NDArray
from skimage.draw import polygon

from src.data_processing.Filters import get_filtered_image_sets
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


def get_image_set(imageType: str, filters=None):
    if filters is None:
        filters = []
    if re.search('[0-9]?[0-9]bin[0-9]?[0-9]max_ones$', imageType) is not None:  # Searching if image type follows the
        # format of 3bin40max_ones
        imageLength = int(re.search(r'^\d+', imageType).group())
        maxOnesPercentage = int(re.search(r'\d+', imageType[2:]).group())
        image_set = get_binary_image_set(imageLength, maxOnesPercentage=maxOnesPercentage)
        logging.info("Applying filters...")
        image_set = get_filtered_image_sets(imageSet=image_set, filters=filters)
    elif re.search('[0-9]?[0-9]bin$', imageType) is not None:  # Searching if the image type follows the format 2bin
        imageLength = int(re.search(r'\d+', imageType).group())
        image_set = get_binary_image_set(imageLength)
        logging.info("Applying filters...")
        image_set = get_filtered_image_sets(imageSet=image_set, filters=filters)
    elif imageType == "triangles":
        image_set = get_triangles_image_set()
        logging.info("Applying filters...")
        image_set = get_filtered_image_sets(imageSet=image_set, filters=filters)
    elif imageType == "quadrilaterals":
        image_set = get_quadrilaterals_image_set()
        logging.info("Applying filters...")
        image_set = get_filtered_image_sets(imageSet=image_set, filters=filters)
    elif re.search(r'randomshapes', imageType) is not None:
        size, border_size, sides, number = parse_shapes_set(imageType, number=True)
        image_set = get_randomized_shapes(size, sides, border_size, number, filters)
    elif re.search(r'shapes', imageType) is not None:
        size, border_size, sides = parse_shapes_set(imageType)
        image_set = []
        for j in sides:
            image_set.extend(get_shapes_set(size, j, border_size).tolist())
        image_set = np.array(image_set)
        logging.info("Applying filters...")
        image_set = get_filtered_image_sets(imageSet=image_set, filters=filters)
    elif re.search('[0-9]?[0-9]island[0-9]?[0-9]max_ones[0-9]?[0-9]images$', imageType) is not None:  # Searching if image type follows the
        # format of 3bin40max_ones
        matches = re.findall(r"\d", imageType)
        imageLength = int(matches[0])
        maxOnesPercentage = int(matches[1])
        numImages = int(matches[2])
        image_set = get_island_image_set(imageLength, maxOnesPercentage, numImages)
        logging.info("Image set generated, applying filters...")
        image_set = get_filtered_image_sets(imageSet=image_set, filters=filters)
    else:
        raise ValueError(imageType + " is not a valid image type")
    return image_set


def get_image_set_size(imageType: str, filters=None) -> int:
    return len(get_image_set(imageType, filters=filters))


def get_island_image_set(imageLength, maxOnesPercentage, numImages):
    """
    :param imageLength: side length of image grid
    :param maxOnesPercentage: Percent of ones
    :param numImages: number of images to generate
    :return: An image set of randomly generated islands with no repeats
    """
    return np.array(grid_creation(imageLength, numImages, int(maxOnesPercentage / 100 * (imageLength ** 2))))


def get_triangles_image_set():
    """
    :return: The image set of 4x4 triangles within an 8x8 matrix
    """
    return get_shapes_set(4, 3, 2)

def get_quadrilaterals_image_set():
    """
    :return: The image set of 4x4 quadrilaterals within an 8x8 matrix
    """
    return get_shapes_set(4, 4, 2)

def parse_shapes_set(imageType: str, number=False):
    """
    Parses the imageType string for an image set containing multiple shapes
    :param imageType: imageType string
    :return: Parameters for the whole image set
    """
    params = imageType.split("_")
    dims = False
    size = 4
    border_size = 2
    sides = []
    for i in range(1, len(params)):
        if params[i] == "dims":
            dims = True
            continue
        elif dims:
            size = int(params[i])
            border_size = int(params[i + 1])
            if number:
                number = int(params[i + 2])
            break
        sides.append(int(params[i]))
    if number:
        return size, border_size, sides, number
    return size, border_size, sides

def get_shapes_set(size: int, sides: int, border_size: int, filters=None):
    if filters is None:
        filters = []
    unique = "unique" in filters
    all_permutations = set()
    image_set = []
    indexes = list(range(0, size ** 2))
    for comb in itertools.permutations(indexes, sides):

        # Check if combination is translationally unique if the option is selected
        if unique and tuple(comb) in all_permutations:
            continue

        image = np.zeros((size, size), dtype=int)
        r = []
        c = []
        for index in comb:
            r.append(index // size)
            c.append(index % size)

        # Check for straight lines
        points = []
        for i in range(0, len(r)):
            points.append((r[i], c[i]))
        collinear = False
        for coords in itertools.combinations(points, 3):
            m1 = (coords[1][1] - coords[0][1]) * (coords[2][0] - coords[1][0])
            m2 = (coords[2][1] - coords[1][1]) * (coords[1][0] - coords[0][0])
            if m1 == m2:
                collinear = True
                break
        if collinear:
            continue

        # Check for intersecting lines within shape
        if sides > 3:
            intersect = False
            lines = []
            for i in range(0, sides):
                start = (r[i % sides], c[i % sides])
                end = (r[(i + 1) % sides], c[(i + 1) % sides])
                lines.append((start, end))
            for pair in itertools.combinations(lines, 2):
                test = is_intersecting(pair[0][0], pair[0][1], pair[1][0], pair[1][1])
                if test:
                    intersect = test
                    break
            if intersect:
                continue

        # Create shape
        rr, cc = polygon(r, c)
        image[rr, cc] = 1
        image = np.pad(image, (border_size, border_size), constant_values=(0, 0))
        image_set.append(image)

        # Accounts for all translationally unique combinations if the option is selected
        if unique:
            all_permutations = add_permutations(all_permutations, comb, size)

    if unique:
        filters.remove("unique")

    image_set = np.array(image_set)
    image_set = np.unique(image_set, axis=0)
    image_set = get_filtered_image_sets(imageSet=image_set, filters=filters)

    return image_set


def cross_test(p1, p2, p3):
    """
    :param p1: Point 1
    :param p2: Point 2
    :param p3: Point 3
    :return: Tests gradient for checking intersections
    """
    return (p3[1] - p1[1]) * (p2[0] - p1[0]) > (p2[1] - p1[1]) * (p3[0] - p1[0])


def is_intersecting(p1, p2, p3, p4):
    """
    Checks if two line segments created using the 4 points intersect in the middle
    :param p1: Point 1
    :param p2: Point 2
    :param p3: Point 3
    :param p4: Point 4
    :return: True if the two lines are intersecting, false if not
    """
    # Check if the segments share a point. If so, they do not intersect in the middle
    if p1 == p4 or p2 == p3:
        return False

    return cross_test(p1, p3, p4) != cross_test(p2, p3, p4) and cross_test(p1, p2, p3) != cross_test(p1, p2, p4)


def get_randomized_shapes(size: int, side_list: list, border_size: int, number: int, filters=None):
    """
    Generates an image set of shapes containing specified number of images
    :param size: Size of the shapes
    :param side_list: Number of sides of the shapes
    :param border_size: Size of the border
    :param number: Number of shapes to generate
    :param filters: Filters to use
    :return: The image set
    """
    random.seed(500)

    image_set = []
    if filters is None:
        filters = []
    unique = "unique" in filters
    if unique:
        filters.remove("unique")
    indexes = list(range(0, size ** 2))
    all_permutations = set()
    while len(image_set) < number:
        random.shuffle(indexes)
        sides = random.choice(side_list)
        comb = indexes[0:sides]
        if tuple(comb) in all_permutations:
            continue

        image = np.zeros((size, size), dtype=int)
        r = []
        c = []
        for index in comb:
            r.append(index // size)
            c.append(index % size)

        # Check for straight lines
        points = []
        for i in range(0, len(r)):
            points.append((r[i], c[i]))
        collinear = False
        for coords in itertools.combinations(points, 3):
            m1 = (coords[1][1] - coords[0][1]) * (coords[2][0] - coords[1][0])
            m2 = (coords[2][1] - coords[1][1]) * (coords[1][0] - coords[0][0])
            if m1 == m2:
                collinear = True
                break
        if collinear:
            continue

        # Check for intersecting lines within shape
        if sides > 3:
            intersect = False
            lines = []
            for i in range(0, sides):
                start = (r[i % sides], c[i % sides])
                end = (r[(i + 1) % sides], c[(i + 1) % sides])
                lines.append((start, end))
            for pair in itertools.combinations(lines, 2):
                test = is_intersecting(pair[0][0], pair[0][1], pair[1][0], pair[1][1])
                if test:
                    intersect = test
                    break
            if intersect:
                continue

        # Create shape
        rr, cc = polygon(r, c)
        image[rr, cc] = 1
        image = np.pad(image, (border_size, border_size), constant_values=(0, 0))
        image_set.append(image.tolist())
        comb = tuple(comb)

        # Accounts for all translationally unique permutations if the option is selected
        # If not, adds the permutation to the set of all permutations to ensure no duplicates
        if unique:
            all_permutations = add_permutations(all_permutations, comb, size)
        else:
            all_permutations.add(comb)
        if len(image_set) == number:
            image_set = get_filtered_image_sets(imageSet=np.array(image_set), filters=filters)
            image_set = np.unique(image_set, axis=0)
            image_set = image_set.tolist()
    image_set = np.array(image_set)
    return image_set


def shift_down(comb: tuple, side: int):
    new = []
    for i in comb:
        j = i + side
        if j >= side ** 2:
            return comb
        new.append(j)
    new = tuple(new)
    return new


def shift_right(comb: tuple, side: int):
    new = []
    for i in comb:
        whole = i // side
        rem = i - (whole * side)
        if rem + 1 >= side:
            return comb
        new.append(whole * side + (rem + 1))
    new = tuple(new)
    return new


def add_permutations(permutations: set, comb_tuple: tuple, size: int):
    for dr in range(size):
        comb_tuple_down = comb_tuple
        for dc in range(size):
            permutations.add(comb_tuple_down)
            comb_tuple_down = shift_down(comb_tuple_down, size)
        comb_tuple = shift_right(comb_tuple, size)
    return permutations

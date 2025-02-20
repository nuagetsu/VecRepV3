import sys
import os
path = os.path.abspath("../VecRepV3") 
sys.path.append(path)
print(path)
import itertools
import numpy as np
import random
from typing import List
from numpy.typing import NDArray
from skimage.draw import polygon
from src.data_processing.Filters import get_filtered_image_sets

IMAGE_TYPES = ["NbinMmax_ones", "Nbin", "triangles", "triangle_mean_subtracted"]

IMAGE_FILTERS = ["unique", "Nmax_ones", "one_island"]

IMAGE_PRODUCT_TYPES = ["ncc", "ncc_scaled"]

EMBEDDING_TYPES = ["pencorr_D"]

dimensions = 32

imageType = "shapes_3_dims_16_8"
filters = ["100max_ones"]


#get_image_set
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

def get_shapes_set(size: int, sides: int, border_size: int, num_images=float('inf'), filters=None):
    """
    Generates a set of shapes, either a fixed number or all permutations.

    :param size: Size of shapes.
    :param sides: Number of sides of shapes.
    :param border_size: Size of the border.
    :param num_images: Number of images to generate (set to float('inf') for all possible permutations).
    :param filters: Filters to apply.
    :return: A set of generated shape images.
    """
    if filters is None:
        filters = []
    
    unique = "unique" in filters
    filter_copy = filters.copy()
    all_permutations = set()
    image_set = []
    
    indexes = list(range(size ** 2))
    count = 0
    
    # If generating all possible permutations
    if num_images == float('inf'):
        iterator = itertools.permutations(indexes, sides)
    else:
        iterator = iter(lambda: random.sample(indexes, sides), None) 

    for comb in iterator:
        if count >= num_images:
            break  

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
        
        # Accounts for all translationally unique combinations if required
        if unique:
            all_permutations = add_permutations(all_permutations, comb, size)
        
        count += 1  # Increment counter

    if unique:
        filter_copy.remove("unique")

    image_set = np.array(image_set)
    image_set = np.unique(image_set, axis=0)
    image_set = get_filtered_image_sets(imageSet=image_set, filters=filter_copy)

    return image_set
    
def get_image_set(imageType: str, max_images=None, filters=None):
    size, border_size, sides = parse_shapes_set(imageType)
    image_set = []
    for j in sides:
        image_set.extend(get_shapes_set(size, j, border_size, max_images).tolist())
    image_set = np.array(image_set)
    image_set = get_filtered_image_sets(imageSet=image_set, filters=filters)
    return image_set

def append_to_npy(imageType: str, filters: List[str], imageSetFilepath: str, max_images=None) -> NDArray:
    new_filtered_images = get_image_set(imageType=imageType, filters=filters, max_images=max_images)
    
    if os.path.exists(imageSetFilepath):
        existing_images = np.load(imageSetFilepath, allow_pickle=True)  
        updated_images = np.concatenate((existing_images, new_filtered_images), axis=0) 
    else:
        updated_images = new_filtered_images 

    np.save(imageSetFilepath, updated_images)

    return updated_images

imageSet = append_to_npy(imageType, filters, 'filtered_images.npy', max_images=2000)

# #np.array the dataset
# imageSet = np.array(imageSet)

# #matrixA then can be generated from any subset
# percentage = 0
# testSize = int(percentage * len(imageSet)) 
# trainingSize = len(imageSet) - testSize
# testSample, trainingSample = SamplingMethod.generate_random_sample(imageSet, testSize, trainingSize)

# sampleName = f"{imageType} {filters} {percentage} sample"

# sampleEstimator = SampleEstimator(sampleName=sampleName, trainingImageSet=trainingSample, embeddingType=embeddingType,
#                                   imageProductType=imageProductType)

import sys
import os
path = os.path.abspath("../VecRepV3") 
sys.path.append(path)
print(path)

import shelve
import itertools
import numpy as np
import random
from typing import List
from numpy.typing import NDArray
from skimage.draw import polygon
from src.data_processing.Filters import get_filtered_image_sets

def parse_shapes_set(imageType: str, number=False):
    """
    Parses the imageType string for an image set containing multiple shapes.
    :param imageType: imageType string (e.g., "shapes_3_dims_16_4")
    :return: Parameters for the whole image set (size, border_size, sides) and optionally number.
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

def generate_shapes(size: int, sides: int, border_size: int, num_images: int, unique: bool=False):
    """
    Generator that yields one valid shape image at a time.
    
    :param size: Size of the shape grid.
    :param sides: Number of sides of the shape.
    :param border_size: Border size to pad the image.
    :param num_images: Maximum number of images to yield.
    :param unique: If True, only yield images from unique (translationally unique) combinations.
    :yield: A valid shape image as a NumPy array.
    """
    indexes = list(range(size ** 2))
    seen = None
    db = None

    try: 
        if unique:
            db = shelve.open('unique_combinations.db', writeback=True)
            seen = db.get('seen', set())
        
        count = 0
        iterator = itertools.permutations(indexes, sides)
        
        for comb in iterator:
            if count >= num_images:
                break
            
            if unique:
                # normalized key to check for uniqueness
                comb_key = tuple(sorted(comb))
                if comb_key in seen:
                    continue
            
            image = np.zeros((size, size), dtype=int)
            r = [index // size for index in comb]
            c = [index % size for index in comb]
            
            # check for collinearity in any three points
            points = list(zip(r, c))
            collinear = False
            for coords in itertools.combinations(points, 3):
                m1 = (coords[1][1] - coords[0][1]) * (coords[2][0] - coords[1][0])
                m2 = (coords[2][1] - coords[1][1]) * (coords[1][0] - coords[0][0])
                if m1 == m2:
                    collinear = True
                    break
            if collinear:
                continue
    
            # for shapes with more than 3 sides, check for intersecting lines
            if sides > 3:
                intersect = False
                # build list of edges for the shape
                lines = [((r[i % sides], c[i % sides]), (r[(i + 1) % sides], c[(i + 1) % sides])) for i in range(sides)]
                for pair in itertools.combinations(lines, 2):
                    if is_intersecting(pair[0][0], pair[0][1], pair[1][0], pair[1][1]):
                        intersect = True
                        break
                if intersect:
                    continue
    
            # fill the polygon
            rr, cc = polygon(r, c)
            image[rr, cc] = 1
            image = np.pad(image, border_size, constant_values=0)
            
            # for unique filtering, need to mark the combination as seen 
            if unique:
                seen.add(comb_key)
                db['seen'] = seen  
                
            count += 1
            yield image
    finally:
        if db:  # Ensure proper cleanup
            db.close()

def get_image_set(imageType: str, max_images=500, filters=None):
    """
    Generates the full image set based on imageType and filters.
    
    :param imageType: String encoding the shapes and dimensions (e.g., "shapes_3_dims_16_4").
    :param max_images: Maximum images per shape (or overall if you prefer).
    :param filters: List of filters (e.g., ["unique"]).
    :return: A NumPy array of filtered shape images.
    """
    size, border_size, sides_list = parse_shapes_set(imageType)
    unique_flag = filters is not None and "unique" in filters

    filters_remaining = filters.copy() if filters else []
    # if unique_flag and "unique" in filters_remaining:
    #     filters_remaining.remove("unique")
    
    all_images = []
    for sides in sides_list:
        for img in generate_shapes(size, sides, border_size, max_images, unique=unique_flag):
            all_images.append(img)
    
    image_set = np.array(all_images)
    print("got filter1")
    image_set = get_filtered_image_sets(imageSet=image_set, filters=filters_remaining)
    
    return image_set

def append_to_npy(imageType: str, filters: list, imageSetFilepath: str, max_images=500):
    """
    Generates new filtered images and appends them to an existing .npy file,
    using a more memory-efficient streaming approach.
    
    :param imageType: The image type string.
    :param filters: List of filters to apply.
    :param imageSetFilepath: Filepath for the .npy file.
    :param max_images: Maximum images to generate per shape type.
    :return: The updated array of images.
    """
    new_images = get_image_set(imageType=imageType, filters=filters, max_images=max_images)
    
    if os.path.exists(imageSetFilepath):
        existing_images = np.load(imageSetFilepath, allow_pickle=True)
        # if large image set, can try saving in batches or using a memmap.
        updated_images = np.concatenate((existing_images, new_images), axis=0)
    else:
        updated_images = new_images
    
    np.save(imageSetFilepath, updated_images)
    
    return updated_images

imageType = "shapes_3_dims_48_4"
filters = ["unique"]
# imageSet = append_to_npy(imageType, filters, 'filtered_images_56x56_eff_200000.npy', max_images=200000)
imageSet = append_to_npy(imageType, filters, '/home/jovyan/VecRepV3/data/train_images_56x56_1.npy', max_images=70000)

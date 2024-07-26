import logging
import os
import sys
from pathlib import Path
from typing import List
from line_profiler import profile

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.data_processing import EmbeddingFunctions
from src.data_processing import ImageGenerators
from src.data_processing import ImageProducts
from src.data_processing.ImageProducts import calculate_image_product_matrix

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

@profile
def generate_filtered_image_set(imageType: str, filters: List[str], imageSetFilepath: str, overwrite=False) -> NDArray:
    """
    :param imageSetFilepath: Place where the image set was previously saved, or the place where the new image set should be saved
    :param overwrite: If true, generates and saves the filtered image set even if it is saved
    :return: An NDArray of filtered image sets as specified by imageType and filters
    Checks if such an image set is already saved in imageSetFilepath. If so it loads the image set
    If not, or if overridden, it uses the ImageGenerator and Filters module to generate a filtered image set
    """
    if not os.path.isfile(imageSetFilepath) or overwrite:
        logging.info("Image set not found/overwrite, generating filtered image set...")
        filteredImageSet = ImageGenerators.get_image_set(imageType=imageType, filters=filters)

        # Creating the directory and saving the image set
        Path(imageSetFilepath).parent.mkdir(parents=True, exist_ok=True)
        np.save(imageSetFilepath, filteredImageSet)
    else:
        filteredImageSet = np.load(imageSetFilepath)
    return filteredImageSet

@profile
def get_image_set_size(imageType: str, filters: List[str], imageSetFilepath:str, overwrite=False):
    """
    :param imageType: Name of the image set
    :param filters: Filters applied to the image set
    :param imageSetFilepath: Place where the image set was previously saved, or the place where the new image set should be saved
    :param overwrite: If true, generates and saves the filtered image set even if it is saved
    :return: An NDArray of filtered image sets as specified by imageType and filters
    Returns the size of an image set
    """
    return len(generate_filtered_image_set(imageType, filters, imageSetFilepath, overwrite=overwrite))


def generate_image_product_matrix(imageSet: NDArray, imageProductType: str, imageProductFilepath: str,
                                  overwrite=False) -> NDArray:
    """
    :param imageProductType: type of image product function to use
    :param imageSet: NDArray of images
    :param imageProductFilepath: Filepath to save/load the image product matrix
    :param overwrite: If true, generates and saves the image product table even if it is saved
    :return: An NDArray which is an image product matrix of the input filtered images and image product
    """
    if not os.path.isfile(imageProductFilepath) or overwrite:
        logging.info("Image product table not found/overwrite, generating image product table...")
        imageProduct = ImageProducts.get_image_product(imageProductType)
        imageProductMatrix = calculate_image_product_matrix(imageSet, imageProduct)

        # Creating the directory and saving the image product matrix
        Path(imageProductFilepath).parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(imageProductFilepath, imageProductMatrix)
    else:
        imageProductMatrix = np.loadtxt(imageProductFilepath)
    return imageProductMatrix


def generate_embedding_matrix(imageProductMatrix, embeddingType, embeddingFilepath, overwrite=False,
                              weight=None):
    """
    :param imageProductMatrix: The image product matrix used to generate the vector embeddings
    :param embeddingType: Method used to generate the vector embeddings
    :param embeddingFilepath: Filepath to save/load the vector embeddings
    :param overwrite: If true, generates and saves the embeddings even if it is saved
    :return: The embedding matrix based on the inputs
    """
    if not os.path.isfile(embeddingFilepath) or overwrite:
        logging.info("Embedding matrix not found/overwrite. Generating embedding matrix...")
        embeddingMatrix = EmbeddingFunctions.get_embedding_matrix(imageProductMatrix, embeddingType,
                                                                  weightMatrix=weight)

        # Creating the directory and saving the embedding matrix
        Path(embeddingFilepath).parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(embeddingFilepath, embeddingMatrix)
    else:
        embeddingMatrix = np.loadtxt(embeddingFilepath)
    return embeddingMatrix

def generate_weighting_matrix(imageProductMatrix, imageSet, weightingType, weightingFilepath,
                              imageProductFilepath, overwrite=False):
    if weightingType == "" or weightingType is None:
        return np.ones_like(imageProductMatrix)
    components = weightingType.split("_factor_")
    if len(components) == 1:
        raise ValueError("Weighting type must indicate weight factor by _factor_[factor]!")
    factor = int(components[1])
    base = components[0]
    if not os.path.isfile(weightingFilepath) or overwrite:
        logging.info("Weighting matrix not found/overwrite. Generating weighting matrix...")
        if base == "copy":
            weightingMatrix = imageProductMatrix ** factor
        else:
            weightingMatrix = generate_image_product_matrix(imageSet, base, imageProductFilepath) ** factor

        # Creating the directory and saving the weighting matrix
        Path(weightingFilepath).parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(weightingFilepath, weightingMatrix)
    else:
        weightingMatrix = np.loadtxt(weightingFilepath)
    return weightingMatrix

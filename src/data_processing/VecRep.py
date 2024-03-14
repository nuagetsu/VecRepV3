import json
import os
import pickle
from pathlib import Path
from typing import Callable
import numpy as np
from numpy.typing import NDArray

from data_processing.ImageProducts import calculate_image_product_matrix
from src.data_processing import ImageGenerators, Filters, ImageProducts, EmbeddingFunctions, FilepathUtils
from visualization import Metrics
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)




def generate_filtered_image_set(*, imageType: str, filters=None, imageSetFilepath: str, overwrite=False) -> NDArray:
    """
    :param overwrite: If true, generates and saves the filtered image set even if it is saved
    :return: An NDArray of filtered image sets as specified
    Checks if such an image set is already saved previously. If so it loads the image set
    If not, or if overridden, it uses the ImageGenerator and Filters module to generate a filtered image set
    """
    if not os.path.isfile(imageSetFilepath) or overwrite:
        imageSet = ImageGenerators.get_image_set(imageType=imageType)
        filteredImageSet = Filters.get_filtered_image_sets(imageSet=imageSet, filters=filters)
        Path(imageSetFilepath).parent.mkdir(parents=True, exist_ok=True)
        np.save(imageSetFilepath, filteredImageSet)
    else:
        filteredImageSet = np.load(imageSetFilepath)
    return filteredImageSet


def generate_image_product_matrix(*, imageSet, imageProductType, imageProductFilepath, overwrite=False) -> NDArray:
    """
    :param imageProductType: type of image product function to use
    :param imageSet: NDArray of images
    :param imageProductFilepath: Filepath to save/load the image product matrix
    :param overwrite: If true, generates and saves the image product table even if it is saved
    :return: An NDArray which is an image product matrix of the input filtered images and image product
    """
    if not os.path.isfile(imageProductFilepath) or overwrite:
        imageProduct = ImageProducts.get_image_product(imageProductType)
        imageProductMatrix = calculate_image_product_matrix(imageSet, imageProduct)
        Path(imageProductFilepath).parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(imageProductFilepath, imageProductMatrix)
    else:
        imageProductMatrix = np.loadtxt(imageProductFilepath)
    return imageProductMatrix


def generate_embedding_matrix(*, imageProductMatrix, embeddingType, embeddingFilepath, overwrite=False):
    """
    :param imageProductMatrix: The image product matrix used to generate the vector embeddings
    :param embeddingType: Method used to generate the vector embeddings
    :param embeddingFilepath: Filepath to save/load the vector embeddings
    :param overwrite: If true, generates and saves the embeddings even if it is saved
    :return: The embedding matrix based on the inputs
    """
    if not os.path.isfile(embeddingFilepath) or overwrite:
        embeddingMatrix = EmbeddingFunctions.get_embedding_matrix(imageProductMatrix=imageProductMatrix,
                                                                  embeddingType=embeddingType)
        Path(embeddingFilepath).parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(embeddingFilepath, embeddingMatrix)
    else:
        embeddingMatrix = np.loadtxt(embeddingFilepath)
    return embeddingMatrix

def generate_plotting_data(*, plottingDataFilepath, imageProductMatrix, embeddingMatrix, imagesFilepath, overwrite=False):
    if not os.path.isfile(plottingDataFilepath) or overwrite:
        plottingData = Metrics.get_plotting_data(imageProductMatrix=imageProductMatrix, embeddingMatrix=embeddingMatrix,
                                                 imagesFilepath=imagesFilepath)
        with open(plottingDataFilepath, 'wb') as f:
            pickle.dump(plottingData, f)
    else:
        with open(plottingDataFilepath, 'rb') as f:
            plottingData = pickle.load(f)

    return plottingData
def get_BF_embeddings(*, imageType: str, filters=None, imageProductType=None, embeddingType=None,
                      overwrite=None) -> NDArray:
    """
    :return: The vector embeddings solved using the brute force method
    """
    if overwrite is None:
        overwrite = {"filter": False, "im_prod": False, "estimate": False, 'plot': False}

    logging.info("Generating filtered images....")
    imageSetFilepath = FilepathUtils.get_image_set_filepath(imageType=imageType, filters=filters)
    imageSet = generate_filtered_image_set(imageType=imageType, filters=filters, imageSetFilepath=imageSetFilepath,
                                           overwrite=overwrite['filter'])
    imageProductFilepath = FilepathUtils.get_image_product_filepath(imageType=imageType, filters=filters,
                                                                    imageProductType=imageProductType)

    logging.info("Generating image product matrix....")
    imageProductMatrix = generate_image_product_matrix(imageSet=imageSet, imageProductType=imageProductType,
                                                       imageProductFilepath=imageProductFilepath,
                                                       overwrite=overwrite['im_prod'])

    logging.info("Generating embeddings....")
    embeddingFilepath = FilepathUtils.get_embedding_matrix_filepath(imageType=imageType, filters=filters,
                                                                    imageProductType=imageProductType,
                                                                    embeddingType=embeddingType)
    embeddingMatrix = generate_embedding_matrix(imageProductMatrix=imageProductMatrix, embeddingType=embeddingType,
                                                embeddingFilepath=embeddingFilepath, overwrite=overwrite['estimate'])
    logging.info("Saving plotting data....")
    plottingDataFilepath = FilepathUtils.get_plotting_data_filepath(imageType=imageType, filters=filters,
                                                                    imageProductType=imageProductType,
                                                                    embeddingType=embeddingType)
    generate_plotting_data(plottingDataFilepath=plottingDataFilepath, imageProductMatrix=imageProductMatrix,
                           embeddingMatrix=embeddingMatrix, imagesFilepath=imageSetFilepath, overwrite=overwrite['plot'])
    return embeddingMatrix

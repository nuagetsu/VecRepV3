import os.path
from pathlib import Path

import numpy as np

import time

from numpy._typing import NDArray

from data_processing import FilepathUtils
from data_processing.ImageProducts import calculate_image_product_vector, get_image_product
from data_processing.VecRep import generate_filtered_image_set, generate_image_product_matrix, \
    generate_embedding_matrix, generate_BF_plotting_data
from helpers.FindingEmbUsingSample import Lagrangian_Method2

import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

def get_unique_sample_name(sampleSize):
    timestr = time.strftime("%Y%m%d-%H%M%S")
    sampleName = str(sampleSize) + "_size_sample_on_" + timestr
    return sampleName


def get_unique_sample_test_name(sampleSize):
    timestr = time.strftime("%Y%m%d-%H%M%S")
    sampleName = str(sampleSize) + "_size_test_sample_on_" + timestr
    return sampleName


class SampleEstimator:
    """
    Initialize this object with the image type, filters, image product and embeddings type which you wish to investigate
    Give the SampleEstimator a unique name which will be used as its directory name
    When loading a SampleEstimator, it will first check if it has been loaded before.
    If so, it will simply load the files, if not it will generate them as required
    """

    def __init__(self, *, sampleName, imageType: str, filters, embeddingType: str,
                 imageProductType: str, overwrite=None, imageSamplesInput=None):
        """
        :param sampleName: Name of the sample (must be unique for each sample) :param imageType: :param filters:
        :param embeddingType: :param imageProductType: :param overwrite: :param imageSamplesInput: An array of images
        which serve as the sample. The embeddings for these images will be calculated. future images can then be made
        into vector embeddings using the sample images as reference
        """
        if overwrite is None:
            overwrite = {"filter": False, "im_prod": False, "estimate": False, 'plot': False, 'sampling': False}

        self.sampleName = sampleName
        self.imageProductType = imageProductType
        self.sampleDirectory = FilepathUtils.get_sample_directory(imageType=imageType, filters=filters,
                                                                  imageProductType=imageProductType,
                                                                  embeddingType=embeddingType, sampleName=self.sampleName)


        logging.info("Loading sample images....")
        sampledImageSetFilepath = FilepathUtils.get_sample_images_filepath(self.sampleDirectory)
        if not os.path.isfile(sampledImageSetFilepath):
            if imageSamplesInput is None:
                raise ValueError("Image samples must be give if sample estimator has not been previously initialized")
            self.sampledImageSet = imageSamplesInput
            Path(sampledImageSetFilepath).parent.mkdir(parents=True, exist_ok=True)
            np.save(sampledImageSetFilepath, self.sampledImageSet)
        else:
            self.sampledImageSet = np.load(sampledImageSetFilepath)

        logging.info("Generating image product matrix....")
        imageProductFilepath = FilepathUtils.get_sample_ipm_filepath(self.sampleDirectory)
        self.imageProductMatrix = generate_image_product_matrix(imageSet=self.sampledImageSet,
                                                                imageProductType=imageProductType,
                                                                imageProductFilepath=imageProductFilepath)
        self.imageProduct = get_image_product(imageProductType)

        logging.info("Generating embeddings....")
        embeddingFilepath = FilepathUtils.get_sample_embedding_filepath(self.sampleDirectory)
        self.embeddingMatrix = generate_embedding_matrix(imageProductMatrix=self.imageProductMatrix,
                                                         embeddingType=embeddingType,
                                                         embeddingFilepath=embeddingFilepath)
        logging.info("Saving plotting data....")
        plottingDataFilepath = FilepathUtils.get_sample_plotting_data_filepath(self.sampleDirectory)
        generate_BF_plotting_data(plottingDataFilepath=plottingDataFilepath, imageProductMatrix=self.imageProductMatrix,
                                  embeddingMatrix=self.embeddingMatrix, imagesFilepath=sampledImageSetFilepath)

    def get_embedding_estimate(self, imageInput)->NDArray:
        """
        :param imageInput: Takes in an image with the same dimensions as images in the image sample
        :return: A vector embedding of the input image generated using the image sample. Method used is by minimizing
        the error between the dot product results and the image product vector.
        """
        imageProductVector = calculate_image_product_vector(imageInput, self.sampledImageSet, self.imageProduct)
        estimateVector = Lagrangian_Method2(self.embeddingMatrix, imageProductVector)[0]
        return estimateVector


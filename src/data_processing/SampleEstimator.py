import logging
import os.path
import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

import FilepathUtils
from ImageProducts import calculate_image_product_vector, get_image_product
from Utilities import generate_image_product_matrix, generate_embedding_matrix
from src.helpers.FindingEmbUsingSample import Lagrangian_Method2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


class SampleEstimator:
    """
    Initialize this object with the sample image set, image product and embeddings type which you wish to investigate
    Give the SampleEstimator a unique name which will be used as its directory name
    When loading a SampleEstimator, it will first check if it's name has been loaded before.
    If so, it will simply load the files, if not it will generate them as required
    """

    def __init__(self, *, sampleName: str, trainingImageSet=None, embeddingType: str,
                 imageProductType: str, overwrite=None):
        """
        :param sampleName: Name of the sample (must be unique for each sample)
        :param embeddingType:
        :param imageProductType:
        :param overwrite:
        :param trainingImageSet: An array of images that is used as the sample set.
        The embeddings for these images will be calculated. future images can then be made
        into vector embeddings using the sample images as reference
        """
        if overwrite is None:
            overwrite = {"imgSet": False, "imgProd": False, "embedding": False}

        self.sampleName = sampleName
        self.imageProductType = imageProductType
        self.sampleDirectory = FilepathUtils.get_sample_directory(self.sampleName)
        self.embeddingType = embeddingType

        # Loading/saving sample image set based on if the file exists
        self.trainingImageSetFilepath = FilepathUtils.get_sample_images_filepath(self.sampleDirectory)

        if not os.path.isfile(self.trainingImageSetFilepath) or overwrite['imgSet']:
            logging.info("Saving sample images....")
            if trainingImageSet is None:
                raise ValueError("Image samples must be give if sample estimator has not been previously initialized")
            self.trainingImageSet = trainingImageSet

            # Making directory if it doesn't exist
            Path(self.trainingImageSetFilepath).parent.mkdir(parents=True, exist_ok=True)
            np.save(self.trainingImageSetFilepath, self.trainingImageSet)
        else:
            logging.info("Previous sample images  loaded....")
            self.trainingImageSet = np.load(self.trainingImageSetFilepath)

        logging.info("Generating image product matrix....")
        imageProductFilepath = FilepathUtils.get_sample_ipm_filepath(self.sampleDirectory)
        self.imageProductMatrix = generate_image_product_matrix(self.trainingImageSet, imageProductType,
                                                                imageProductFilepath, overwrite=overwrite['imgProd'])
        self.imageProduct = get_image_product(imageProductType)

        logging.info("Generating embeddings....")
        embeddingFilepath = FilepathUtils.get_sample_embedding_filepath(self.sampleDirectory)
        self.embeddingMatrix = generate_embedding_matrix(self.imageProductMatrix, embeddingType, embeddingFilepath,
                                                         overwrite=overwrite['embedding'])

    def get_embedding_estimate(self, imageInput) -> NDArray:
        """
        :param imageInput: Takes in an image with the same dimensions as images in the image sample
        :return: A vector embedding of the input image generated using the image sample. Method used is by minimizing
        the error between the dot product results and the image product vector.
        """
        imageProductVector = calculate_image_product_vector(imageInput, self.trainingImageSet, self.imageProduct)
        estimateVector = Lagrangian_Method2(self.embeddingMatrix, imageProductVector)[0]
        return estimateVector

    def to_string(self):
        return self.sampleName + ", " + self.imageProductType + ", " + self.embeddingType

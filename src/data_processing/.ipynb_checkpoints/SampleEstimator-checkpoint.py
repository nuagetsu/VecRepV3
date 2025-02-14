import logging
import os.path
import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from src.helpers import FilepathUtils
from src.data_processing.ImageProducts import get_image_product
from src.data_processing.Utilities import (generate_image_product_matrix, generate_embedding_matrix,
                                           generate_weighting_matrix)
from src.helpers.FindingEmbUsingSample import get_embedding_estimate

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
                 imageProductType: str, overwrite=None, parentImageSet="uncategorized", weight=None):
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
        self.sampleDirectory = FilepathUtils.get_sample_directory(self.sampleName, category=parentImageSet)
        self.embeddingType = embeddingType
        self.imageProduct = get_image_product(imageProductType)

        # Loading/saving sample image set based on if the file exists
        self.trainingImageSetFilepath = FilepathUtils.get_sample_images_filepath(self.sampleDirectory)

        if not os.path.isfile(self.trainingImageSetFilepath) or overwrite['imgSet']:
            logging.info("Saving sample images....")
            if trainingImageSet is None:
                raise ValueError("Image samples must be given if sample estimator has not been previously initialized")
            self.trainingImageSet = trainingImageSet

            # Making directory if it doesn't exist
            Path(self.trainingImageSetFilepath).parent.mkdir(parents=True, exist_ok=True)
            np.save(self.trainingImageSetFilepath, self.trainingImageSet)
        else:
            logging.info("Previous sample images loaded....")
            self.trainingImageSet = np.load(self.trainingImageSetFilepath)

        self.imageProductFilepath = os.path.join(self.sampleDirectory, imageProductType)
        if not os.path.isfile(self.imageProductFilepath):
            Path(self.imageProductFilepath).parent.mkdir(parents=True, exist_ok=True)

        logging.info("Generating image product matrix....")
        imageProductMatrixFilepath = FilepathUtils.get_sample_ipm_filepath(self.imageProductFilepath)
        self.imageProductMatrix = generate_image_product_matrix(self.trainingImageSet, imageProductType,
                                                                imageProductMatrixFilepath, overwrite=overwrite['imgProd'])

        if weight is None or weight == "":
            weight = ""
        weightingFilepath = FilepathUtils.get_sample_weighting_filepath(self.sampleDirectory, weight, copy=imageProductType)

        weightMatrix = generate_weighting_matrix(self.imageProductMatrix, self.trainingImageSet, weight, weightingFilepath,
                                                           imageProductMatrixFilepath, overwrite['imgProd'])


        self.embeddingFilepath = Path(FilepathUtils.get_sample_embedding_matrix_filepath(
            embeddingType, self.imageProductFilepath, weight=weight)).parent
        if not os.path.isfile(self.embeddingFilepath):
            Path(self.embeddingFilepath).parent.mkdir(parents=True, exist_ok=True)
        logging.info("Generating embeddings....")
        embeddingMatrixFilepath = FilepathUtils.get_sample_embedding_matrix_filepath(embeddingType,
                                                                                     self.imageProductFilepath,
                                                                                     weight=weight)
        self.embeddingMatrix = generate_embedding_matrix(self.imageProductMatrix, embeddingType, embeddingMatrixFilepath,
                                                         overwrite=overwrite['embedding'], weight=weightMatrix)

    def get_embedding_estimate(self, imageInput) -> NDArray:
        """
        :param imageInput: Takes in an image with the same dimensions as images in the image sample
        :return: A vector embedding of the input image generated using the image sample. Method used is by minimizing
        the error between the dot product results and the image product vector.
        """
        return get_embedding_estimate(imageInput, self.trainingImageSet, self.imageProductType, self.embeddingMatrix)

    def to_string(self):
        return self.sampleName + ", " + self.imageProductType + ", " + self.embeddingType

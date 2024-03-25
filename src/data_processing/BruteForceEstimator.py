from typing import List

import numpy as np

import src.data_processing.FilepathUtils as fpUtils
import src.data_processing.Utilities as utils
from src.data_processing.EmbeddingFunctions import get_eig_for_symmetric
import src.visualization.Metrics as metrics


class BruteForceEstimator:
    def __init__(self, *, imageType: str, filters: List[str], imageProductType: str, embeddingType: str,
                 overwrite=None):
        self.imageType = imageType
        self.filters = filters
        self.imageProductType = imageProductType
        self.embeddingType = embeddingType
        if overwrite is None:
            overwrite = {"imgSet": False, "imgProd": False, "embedding": False}

        # Getting filepaths
        self.imageFilepath = fpUtils.get_image_set_filepath(imageType, filters)
        self.imgProductFilepath = fpUtils.get_image_product_filepath(imageType, filters, imageProductType)
        self.embeddingFilepath = fpUtils.get_embedding_matrix_filepath(imageType, filters, imageProductType,
                                                                       embeddingType)

        # Generating image set and matrix G, G' and A
        self.imageSet = utils.generate_filtered_image_set(imageType, filters, self.imageFilepath, overwrite['imgSet'])
        self.matrixG = utils.generate_image_product_matrix(self.imageSet, imageProductType, self.imgProductFilepath,
                                                           overwrite['imgProd'])
        self.matrixA = utils.generate_embedding_matrix(self.matrixG, embeddingType, self.embeddingFilepath,
                                                       overwrite["embedding"])
        self.matrixGprime = np.matmul(self.matrixA.T, self.matrixA)

    @property
    def initialEigenvalues(self):
        initialEigenvalues, vec = get_eig_for_symmetric(self.matrixG)
        return initialEigenvalues

    @property
    def finalEigenvalues(self):
        finalEigenvalues, vec = get_eig_for_symmetric(self.matrixGprime)
        return finalEigenvalues

    @property
    def frobDistance(self):
        return metrics.get_frob_distance(self.matrixG, self.matrixGprime)

    @property
    def aveFrobDistance(self):
        return self.frobDistance / (len(self.imageSet) ** 2)

    @property
    def maxDifference(self):
        return np.max(np.abs(self.matrixG - self.matrixGprime))

    def to_string(self):
        filterString = "["
        for f in self.filters:
            filterString = filterString + f + ", "
        filterString = filterString[:-2] + "]"
        return self.imageType + ", " + filterString + ", " + self.imageProductType + ", " + self.embeddingType

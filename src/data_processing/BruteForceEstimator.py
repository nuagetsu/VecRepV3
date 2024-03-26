from typing import List

import numpy as np

import src.data_processing.FilepathUtils as fpUtils
import src.data_processing.Utilities as utils
from src.data_processing.EmbeddingFunctions import get_eig_for_symmetric
import src.visualization.Metrics as metrics
from src.data_processing.TestableEstimator import TestableEstimator


class BruteForceTestableEstimator(TestableEstimator):
    def __init__(self, *, imageType: str, filters: List[str], imageProductType: str, embeddingType: str,
                 overwrite=None):
        self.imageType = imageType
        self.filters = filters
        if overwrite is None:
            overwrite = {"imgSet": False, "imgProd": False, "embedding": False}

        # Getting filepaths
        imageFilepath = fpUtils.get_image_set_filepath(imageType, filters)
        self.imgProductFilepath = fpUtils.get_image_product_filepath(imageType, filters, imageProductType)
        self.embeddingFilepath = fpUtils.get_embedding_matrix_filepath(imageType, filters, imageProductType,
                                                                       embeddingType)

        # Generating image set and matrix G, G' and A
        imageSet = utils.generate_filtered_image_set(imageType, filters, imageFilepath, overwrite['imgSet'])
        matrixG = utils.generate_image_product_matrix(imageSet, imageProductType, self.imgProductFilepath,
                                                      overwrite['imgProd'])
        matrixA = utils.generate_embedding_matrix(matrixG, embeddingType, self.embeddingFilepath,
                                                  overwrite["embedding"])

        super().__init__(imageSet, imageFilepath, imageProductType, embeddingType, matrixA, matrixG)

    def to_string(self):
        filterString = "["
        for f in self.filters:
            filterString = filterString + f + ", "
        filterString = filterString[:-2] + "]"
        return self.imageType + ", " + filterString + ", " + self.imageProductType + ", " + self.embeddingType

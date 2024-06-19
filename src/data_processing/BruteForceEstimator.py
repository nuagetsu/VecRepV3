from typing import List

import helpers.FilepathUtils as fpUtils
import src.data_processing.Utilities as utils
from src.data_processing.TestableEstimator import TestableEstimator


class BruteForceEstimator(TestableEstimator):
    def __init__(self, *, imageType: str, filters: List[str], imageProductType: str, embeddingType: str,
                 weightType=None, overwrite=None, weightMatrix=None):
        self.imageType = imageType
        self.filters = filters
        if overwrite is None:
            overwrite = {"imgSet": False, "imgProd": False, "embedding": False}
        if weightType is None:
            weightType = ""

        # Getting filepaths
        imageFilepath = fpUtils.get_image_set_filepath(imageType, filters)
        self.imgProductFilepath = fpUtils.get_image_product_filepath(imageType, filters, imageProductType)
        self.embeddingFilepath = fpUtils.get_embedding_matrix_filepath(imageType, filters, imageProductType,
                                                                       embeddingType, weight=weightType)
        self.weightingFilepath = fpUtils.get_weighting_matrix_filepath(imageType, filters, weightType,
                                                                       copy=imageProductType)

        # Generating image set and matrix G, G' and A
        imageSet = utils.generate_filtered_image_set(imageType, filters, imageFilepath, overwrite['imgSet'])
        matrixG = utils.generate_image_product_matrix(imageSet, imageProductType, self.imgProductFilepath,
                                                      overwrite['imgProd'])
        if weightMatrix is None:
            weightMatrix = utils.generate_weighting_matrix(matrixG, imageSet, weightType, self.weightingFilepath,
                                                           self.imgProductFilepath, overwrite['imgProd'])

        matrixA = utils.generate_embedding_matrix(matrixG, embeddingType, self.embeddingFilepath,
                                                  overwrite["embedding"], weight=weightMatrix)

        super().__init__(imageSet, imageFilepath, imageProductType, embeddingType, matrixA, matrixG)

    def to_string(self):
        filterString = "["
        for f in self.filters:
            filterString = filterString + f + ", "
        if self.filters:
            filterString = filterString[:-2] + "]"
        else:
            filterString = filterString + "]"
        return self.imageType + ", " + filterString + ", " + self.imageProductType + ", " + self.embeddingType

import numpy as np
from numpy.typing import NDArray

import src.visualization.Metrics as metrics
from src.data_processing.EmbeddingFunctions import get_eig_for_symmetric


class TestableEstimator:
    def __init__(self, imageSet: NDArray, imageFilepath: str, imageProductType: str, embeddingType: str, matrixA: NDArray, matrixG: NDArray):
        self.imageSet = imageSet
        self.imageFilepath = imageFilepath
        self.imageProductType = imageProductType
        self.embeddingType = embeddingType
        self.matrixA = matrixA
        self.matrixG = matrixG
        self.matrixGprime = np.matmul(matrixA.T, matrixA)
        self.initialEigenvalues, vec = get_eig_for_symmetric(self.matrixG)
        self.finalEigenvalues, vec = get_eig_for_symmetric(self.matrixGprime)
        self.frobDistance = metrics.get_frob_distance(self.matrixG, self.matrixGprime)
        self.aveFrobDistance = self.frobDistance / (len(self.matrixG) ** 2)
        self.maxDifference = np.max(np.abs(self.matrixG - self.matrixGprime))

    def to_string(self):
        return "Default Name"

import numpy as np
from numpy.typing import NDArray

import src.visualization.Metrics as metrics
from src.data_processing.EmbeddingFunctions import get_eig_for_symmetric


class TestableEstimator:
    def __init__(self, imageSet: NDArray, imageFilepath: str):
        self.imageSet = imageSet
        self.imageFilepath = imageFilepath

    def to_string(self):
        return "Default Name"

from typing import List
from line_profiler import profile

import src.helpers.FilepathUtils as fpUtils
import src.data_processing.Utilities as utils
from src.data_processing.ModifiedTestableEstimator import TestableEstimator


class BruteForceEstimator(TestableEstimator):
    @profile
    def __init__(self, *, imageType: str, filters: List[str], overwrite=None):
        self.imageType = imageType
        self.filters = filters
        if overwrite is None:
            overwrite = {"imgSet": False}

        imageFilepath = fpUtils.get_image_set_filepath(imageType, filters)
        
        imageSet = utils.generate_filtered_image_set(imageType, filters, imageFilepath, overwrite['imgSet'])

        super().__init__(imageSet, imageFilepath)

    def to_string(self):
        filterString = "["
        for f in self.filters:
            filterString = filterString + f + ", "
        if self.filters:
            filterString = filterString[:-2] + "]"
        else:
            filterString = filterString + "]"
        return self.imageType + ", " + filterString + ", " + self.imageProductType + ", " + self.embeddingType

import logging
import os.path
from pathlib import Path

import numpy as np
from numpy._typing import NDArray

import src.data_processing.Utilities as utils
import src.helpers.FilepathUtils as fputils

from src.data_processing.ImageProducts import calculate_image_product_matrix
from src.data_processing.TestableEstimator import TestableEstimator
from src.helpers.FindingEmbUsingSample import get_embedding_estimate


class ShapeMatchingTester:

    def __init__(self, *, training_estimator: TestableEstimator, matching_set_name: str,
                 overwrite=None, matching_images=None,  matching_set_filters=None):
        if overwrite is None:
            overwrite = {"imgSet": False, "imgProd": False, "embedding": False}
        self.training_estimator = training_estimator
        self.matching_set_name = matching_set_name
        self.matching_set_filepath, self.matching_images = self.get_matching_set(matching_set_name,
                                                                                 matching_set_filters=matching_set_filters,
                                                                                 matching_images=matching_images,
                                                                                 overwrite=overwrite["imgSet"])
        """
        Add: Embedding Set: Set of all embeddings? Maybe dictionary?
             Find closest images for each? Or relegate to method
             Clustering??? Find out if image embeddings are evenly spaced? How to cluster against cosine similarity?
             Can we categorise shapes?
             Compare actual closest shapes to generated closest shapes from embedding
        """

    def match_shapes(self, input_image: NDArray, k=5):
        embedding = get_embedding_estimate(input_image, self.training_estimator.imageSet, self.training_estimator.imageProductType,
                                           self.training_estimator.matrixA)
        b = [np.dot(np.atleast_1d(embedding), np.atleast_1d(x)) for x in self.training_estimator.matrixA.T]
        image_set = self.training_estimator.imageSet
        k += 1
        imgProd_max_index = np.argpartition(b, -k)[-k:]
        nearest_images = image_set[imgProd_max_index]
        return nearest_images




    def get_matching_set(self, matching_set_name: str, matching_set_filters=None, matching_images=None, overwrite=False):
        filename = matching_set_name
        for i in matching_set_filters:
            filename += i
        matching_set_filepath = fputils.get_matching_sample_filepath(matching_set_name)
        if matching_set_name == "training":
            matching_set_filepath = self.training_estimator.imageFilepath
            matching_image_set = self.training_estimator.imageSet
        elif matching_images is not None:
            matching_image_set = matching_images
            if not os.path.isfile(matching_set_filepath) or overwrite:
                logging.info("Saving matching sample images....")
                Path(matching_set_filepath).parent.mkdir(parents=True, exist_ok=True)
                np.save(matching_set_filepath, matching_image_set)
        elif not os.path.isfile(matching_set_filepath) or overwrite:
            logging.info("Saving matching sample images....")
            Path(matching_set_filepath).parent.mkdir(parents=True, exist_ok=True)
            matching_image_set = utils.generate_filtered_image_set(matching_set_name, matching_set_filters,
                                                                   matching_set_filepath)
            np.save(matching_set_filepath, matching_image_set)
        else:
            matching_image_set = np.load(matching_set_filepath)
        return matching_set_filepath, matching_image_set





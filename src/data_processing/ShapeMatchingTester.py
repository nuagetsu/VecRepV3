import logging
import os.path
from pathlib import Path

import numpy as np
from numpy._typing import NDArray

import src.data_processing.Utilities as utils
import src.helpers.FilepathUtils as fputils

from src.data_processing.ImageProducts import calculate_image_product_matrix
from src.data_processing.Filters import apply_translationally_unique_filter
from src.data_processing.ImageProducts import get_image_product
from src.data_processing.TestableEstimator import TestableEstimator
from src.helpers.FindingEmbUsingSample import get_embedding_estimate


def get_matching_set(matching_set_name: str, matching_set_filters=None, matching_images=None,
                     overwrite=False):
    filename = matching_set_name
    for i in matching_set_filters:
        filename += "-" + i
    matching_set_filepath = fputils.get_matching_sample_filepath(filename)
    if matching_images is not None:
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


class ShapeMatchingTester:

    def __init__(self, *, training_estimator: TestableEstimator, matching_set_name: str,
                 overwrite=None, matching_images=None, matching_set_filters=None):
        if overwrite is None:
            overwrite = {"imgSet": False, "imgProd": False, "embedding": False}
        self.training_estimator = training_estimator
        self.matching_set_name = matching_set_name
        self.matching_set_filepath, self.matching_set = get_matching_set(matching_set_name,
                                                                         matching_set_filters=matching_set_filters,
                                                                         matching_images=matching_images,
                                                                         overwrite=overwrite["imgSet"])

        logging.info("Initializing...")
        self.full_image_set_filepath = fputils.get_full_matching_image_set_filepath(self.matching_set_filepath,
                                                                                    os.path.basename(os.path.split(
                                                                                        self.training_estimator.
                                                                                        imageFilepath)[0]))

        # Create the full image set of training images and test images
        if not os.path.isfile(self.full_image_set_filepath) or overwrite["imgSet"]:
            # Combine image sets
            logging.info("Combining image sets...")
            full_image_set = []
            full_image_set.extend(self.training_estimator.imageSet.tolist())
            full_image_set.extend(self.matching_set)
            logging.info("Applying filters...")
            self.full_image_set = apply_translationally_unique_filter(np.array(full_image_set))
            Path(self.full_image_set_filepath).parent.mkdir(parents=True, exist_ok=True)
            np.save(self.full_image_set_filepath, self.full_image_set)
        else:
            logging.info("Loading combined image set...")
            self.full_image_set = np.load(self.full_image_set_filepath)

        self.embedding_filepath = fputils.get_matching_embeddings_filepath(
            os.path.split(self.full_image_set_filepath)[0],
            self.training_estimator.imageProductType,
            self.training_estimator.embeddingType)
        self.image_dict = {}

        # Find embeddings of all images. If image is in training set, using brute force embeddings. Else,
        # use embeddings found with the Lagrangian method.
        if not os.path.isfile(self.embedding_filepath) or overwrite["imgSet"]:
            logging.info("Generating embeddings...")
            total_size = len(self.full_image_set)
            embedding_matrix = []
            for index, image in enumerate(self.full_image_set):
                logging.info("Generating embedding " + str(index + 1) + "/" + str(total_size))
                image_list = self.training_estimator.imageSet.tolist()
                if image.tolist() in image_list:
                    x = image_list.index(image.tolist())
                    embedding = self.training_estimator.matrixA.T[x]
                else:
                    embedding = get_embedding_estimate(image, self.training_estimator.imageSet,
                                                       self.training_estimator.imageProductType,
                                                       self.training_estimator.matrixA)
                self.image_dict[index] = {"image": image, "embedding": embedding}
                embedding_matrix.append(embedding)
            self.embedding_set = np.array(embedding_matrix)
            Path(self.embedding_filepath).parent.mkdir(parents=True, exist_ok=True)
            np.save(self.embedding_filepath, embedding_matrix)
        else:
            logging.info("Loading embeddings...")
            self.embedding_set = np.load(self.embedding_filepath)
            for index, image in enumerate(self.full_image_set):
                embedding = self.embedding_set[index]
                self.image_dict[index] = {"image": image, "embedding": embedding}
        """
        Add: Embedding Set: Set of all embeddings? Maybe dictionary? Done
             Find closest images for each? Or relegate to method Done
             Match images not in matching set Done
             Repeated random matching Done
             Add more logging Done
             Add actual matches Done
             Clustering??? Find out if image embeddings are evenly spaced? How to cluster against cosine similarity?
             Can we categorise shapes?
             Compare actual closest shapes to generated closest shapes from embedding Done
        """

    def match_shapes(self, input_image: NDArray, k=5):
        # Search for actual nearest images
        image_product = get_image_product(self.training_estimator.imageProductType)
        c = [image_product(input_image, self.image_dict[image]["image"]) for image in self.image_dict]
        k += 1
        actual_img_prod_max_index = np.argpartition(c, -k)[-k:]
        actual_img_prod_max_index = actual_img_prod_max_index[::-1]
        actual_nearest_images = self.full_image_set[actual_img_prod_max_index]

        # Search for nearest images using embeddings
        if input_image.tolist() in self.full_image_set.tolist():
            index = self.full_image_set.tolist().index(input_image.tolist())
            embedding = self.image_dict[index]["embedding"]
            if "nearest_images" not in self.image_dict[index]:
                b = [np.dot(np.atleast_1d(embedding), np.atleast_1d(self.image_dict[x]["embedding"])) for x in
                     self.image_dict]
                img_prod_max_index = np.argpartition(b, -k)[-k:]
                img_prod_max_index = img_prod_max_index[::-1]
                nearest_images = self.full_image_set[img_prod_max_index]
                self.image_dict[index]["nearest_images"] = nearest_images
            else:
                nearest_images = self.image_dict[index]["nearest_images"]
        else:
            embedding = get_embedding_estimate(input_image, self.training_estimator.imageSet,
                                               self.training_estimator.imageProductType,
                                               self.training_estimator.matrixA)
            b = [np.dot(np.atleast_1d(embedding), np.atleast_1d(self.image_dict[x]["embedding"])) for x in
                 self.image_dict]
            img_prod_max_index = np.argpartition(b, -k)[-k:]
            img_prod_max_index = img_prod_max_index[::-1]
            nearest_images = self.full_image_set[img_prod_max_index]
        return nearest_images, actual_nearest_images

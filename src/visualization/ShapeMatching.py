import logging
import math
import random
import sys

import matplotlib.pyplot as plt
import numpy as np

import visualization.Metrics as metrics
from data_processing.ShapeMatchingTester import ShapeMatchingTester
from src.data_processing.BruteForceEstimator import BruteForceEstimator
from src.data_processing.TestableEstimator import TestableEstimator
from visualization import GraphEstimates

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

def match_random_shape(*, training_image_type, training_filters, imageProductType, embeddingType, weight,
                       test_image_type, test_filters, overwrite=None):

    estimator = BruteForceEstimator(imageType=training_image_type, filters=training_filters,
                                    imageProductType=imageProductType, embeddingType=embeddingType, weightType=weight,
                                    overwrite=overwrite)

    tester = ShapeMatchingTester(training_estimator=estimator, matching_set_name=test_image_type,
                                 matching_set_filters=test_filters)

    image = random.choice(tester.full_image_set)
    nearest_images = tester.match_shapes(image)

    fig, axes = plt.subplots(6, 2)

    axes[0][0].imshow(image)
    for index, im in enumerate(nearest_images):
        axes[index][1].imshow(im)

    plt.show()

def match_shapes_with_index(*, training_image_type, training_filters, imageProductType, embeddingType, weight,
                       test_image_type, test_filters, overwrite=None):

    estimator = BruteForceEstimator(imageType=training_image_type, filters=training_filters,
                                    imageProductType=imageProductType, embeddingType=embeddingType, weightType=weight,
                                    overwrite=overwrite)

    tester = ShapeMatchingTester(training_estimator=estimator, matching_set_name=test_image_type,
                                 matching_set_filters=test_filters)

    while True:
        index = input("Enter an index to match its image\nEnter 'random' to match a random image\nEnter 'quit' to " +
                      "exit\nIndex:")

        if index == "random":
            image = random.choice(tester.full_image_set)
        elif index == "quit":
            break
        else:
            image = tester.full_image_set[int(index)]

        nearest_images = tester.match_shapes(image)

        fig, axes = plt.subplots(6, 2)

        axes[0][0].imshow(image)
        for index, im in enumerate(nearest_images):
            axes[index][1].imshow(im)

        plt.show()

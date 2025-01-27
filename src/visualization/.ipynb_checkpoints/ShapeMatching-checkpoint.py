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
                       test_image_type, test_filters, overwrite=None, k=5):

    estimator = BruteForceEstimator(imageType=training_image_type, filters=training_filters,
                                    imageProductType=imageProductType, embeddingType=embeddingType, weightType=weight,
                                    overwrite=overwrite)

    tester = ShapeMatchingTester(training_estimator=estimator, matching_set_name=test_image_type,
                                 matching_set_filters=test_filters)

    image = random.choice(tester.full_image_set)
    nearest_images, actual_nearest_images = tester.match_shapes(image, k=k)

    fig, axes = plt.subplots(k + 1, 3)

    # Show image to match
    axes[0][0].axis('off')
    axes[1][0].axis('off')
    axes[2][0].axis('off')
    axes[3][0].imshow(image)
    axes[3][0].set_title("Image to Match")
    axes[4][0].axis('off')
    axes[5][0].axis('off')

    # Show nearest images found from embeddings
    for index, im in enumerate(nearest_images):
        axes[index][1].imshow(im)
        if index == 0:
            axes[index][1].set_title("Nearest Images from Embeddings")

    # Show nearest images found by taking image product with all images
    for index, im in enumerate(actual_nearest_images):
        axes[index][2].imshow(im)
        if index == 0:
            axes[index][2].set_title("Nearest Images from Image Products")

    plt.show()


def match_shapes_with_index(*, training_image_type, training_filters, imageProductType, embeddingType, weight,
                       test_image_type, test_filters, overwrite=None, k=5):

    estimator = BruteForceEstimator(imageType=training_image_type, filters=training_filters,
                                    imageProductType=imageProductType, embeddingType=embeddingType, weightType=weight,
                                    overwrite=overwrite)

    tester = ShapeMatchingTester(training_estimator=estimator, matching_set_name=test_image_type,
                                 matching_set_filters=test_filters)

    while True:
        index = input("Enter an index to match its image\nEnter 'random' to match a random image\nEnter 'quit' to " +
                      "exit\nIndex:")

        if index == "random":
            random_choice = random.randint(0, len(tester.full_image_set))
            image = tester.full_image_set[random_choice]
            logging.info("Matching with image of index " + str(random_choice))
        elif index == "quit":
            break
        else:
            logging.info("Matching with image of index " + index)
            image = tester.full_image_set[int(index)]

        nearest_images, actual_nearest_images = tester.match_shapes(image, k=k)
        largest = max(len(nearest_images), len(actual_nearest_images))

        fig, axes = plt.subplots(largest, 3)

        # Show image to match
        for index in range(0, largest):
            if index == (largest // 2) - 1:
                axes[index][0].imshow(image)
                axes[index][0].set_title("Image to Match")
            else:
                axes[index][0].axis('off')

        # Show nearest images found from embeddings
        for index, im in enumerate(nearest_images):
            axes[index][1].imshow(im)
            if index == 0:
                axes[index][1].set_title("Nearest Images from Embeddings")

        # Show nearest images found by taking image product with all images
        for index, im in enumerate(actual_nearest_images):
            axes[index][2].imshow(im)
            if index == 0:
                axes[index][2].set_title("Nearest Images from Image Products")

        plt.show()

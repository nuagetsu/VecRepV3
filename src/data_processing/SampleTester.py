import os.path
import pickle
from pathlib import Path

import numpy as np
from data_processing import SampleEstimator, FilepathUtils
from data_processing.SampleEstimator import SampleEstimator
from src.data_processing.ImageProducts import calculate_image_product_matrix, get_image_product
import logging

class SampleTester:
    """
    An object which contains an array of test images
    and a SampleEstimator object (which itself has its own set of sample images)
    The results of a sample tester is saved to a directory (determined by the testName)
    If the SampleTest has been previously initialized, it will load the relevant files
    The files of SampleTester are saved in a directory, the directory contains the images, embeddings and plotting data
    """

    def __init__(self, *, testImages=None, sampleEstimator: SampleEstimator, testName: str, overwrite=None):
        """
        :param testImages: Array of images used as the test image set
        :param sampleEstimator: Sample estimator which will be tested
        :param testName: Unique name of a test. Used to create the directory.
        Loads the required variables if the test has been run before.
        Else it generates and saves the variables
        """
        if overwrite is None:
            overwrite = {"imgSet": False, "imgProd": False, "embedding": False}

        self.testName = testName
        self.sampleEstimator = sampleEstimator

        # Loading/saving test images
        logging.info("Loading test images...")
        self.testImagesFilepath = FilepathUtils.get_test_images_filepath(sampleEstimator.sampleDirectory, testName)
        if not os.path.isfile(self.testImagesFilepath) or overwrite['imgSet']:
            if testImages is None:
                raise ValueError("Test images must be given, unless test has already been previously created")
            logging.info("Test images not found, saving test images...")
            Path(self.testImagesFilepath).parent.mkdir(parents=True, exist_ok=True)
            np.save(self.testImagesFilepath, testImages)
            self.testImages = testImages
        else:
            self.testImages = np.load(self.testImagesFilepath)

        logging.info("Generating test embeddings...")
        self.testEmbeddingsFilepath = FilepathUtils.get_test_embeddings_filepath(sampleEstimator.sampleDirectory, testName)
        if not os.path.isfile(self.testEmbeddingsFilepath) or overwrite['embedding']:
            logging.info("Embeddings not found, calculating embeddings images...")
            testEmbeddingMatrix = []
            for image in self.testImages:
                testEmbeddingMatrix.append(sampleEstimator.get_embedding_estimate(image))
            testEmbeddingMatrix = np.array(testEmbeddingMatrix)
            testEmbeddingMatrix = testEmbeddingMatrix.T  # Embedding matrix has vectors in columns instead of rows

            # Creating directory if it doesn't exist
            Path(self.testEmbeddingsFilepath).parent.mkdir(parents=True, exist_ok=True)
            np.savetxt(self.testEmbeddingsFilepath, testEmbeddingMatrix)
            self.testEmbeddingMatrix = testEmbeddingMatrix
        else:
            self.testEmbeddingMatrix = np.loadtxt(self.testEmbeddingsFilepath)

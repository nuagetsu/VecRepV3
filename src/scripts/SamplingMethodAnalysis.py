import os.path
import pickle
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from numpy._typing import NDArray

from data_processing import SamplingMethod, FilepathUtils
from data_processing.SamplingMethod import SampleEstimator
from data_processing.VecRep import generate_filtered_image_set
from data_processing.FilepathUtils import get_image_set_filepath
from src.data_processing.ImageProducts import calculate_image_product_matrix, get_image_product
import logging
import sys

from visualization import GraphEstimates
from visualization.Metrics import get_sample_plotting_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

def get_random_image_sample(*, imageType: str, filters=None, nSamples: int) -> NDArray:
    """
    :param imageType:
    :param filters:
    :param nSamples: Number of image samples to take
    :return: An array of length nSample, containing images which are randomly selected from the imageType and filters specified
    """
    imageSetFilepath = get_image_set_filepath(imageType=imageType, filters=filters)
    imageSet = generate_filtered_image_set(imageType=imageType, filters=filters, imageSetFilepath=imageSetFilepath)

    # Sampling the image set
    if sampleSize >= len(imageSet):
        raise ValueError("n must be less than the length of the array.")
    random_indices = np.random.choice(len(imageSet), size=nSamples, replace=False)
    sampledImageSet = imageSet[random_indices]
    return sampledImageSet


class SampleTester:
    """
    An object which contains an array of test images
    and a SampleEstimator object (which itself has its own set of sample images)
    The results of a sample tester is saved to a directory (determined by the testName)
    If the SampleTest has been previously initialized, it will load the relevant files
    The files of SampleTester are saved in a directory, the directory contains the images, embeddings and plotting data
    """

    def __init__(self, *, testImages=None, sampleEstimator: SampleEstimator, sampleDir: str, testName: str):
        logging.info("Loading test images...")
        testImagesFilepath = FilepathUtils.get_sample_test_images_filepath(sampleDir, testName)
        if not os.path.isfile(testImagesFilepath):
            if testImages is None:
                raise ValueError("Test images must be given, unless test has already been previously created")
            Path(testImagesFilepath).parent.mkdir(parents=True, exist_ok=True)
            np.save(testImagesFilepath, testImages)
            self.testImages = testImages
        else:
            self.testImages = np.load(testImagesFilepath)

        logging.info("Generating test embeddings...")
        testSampleEmbeddingsFilepath = FilepathUtils.get_sample_test_embeddings_filepath(sampleDir, testName)
        if not os.path.isfile(testSampleEmbeddingsFilepath):
            testSampleEmbeddingMatrix = []
            for image in self.testImages:
                testSampleEmbeddingMatrix.append(sampleEstimator.get_embedding_estimate(image))
            testSampleEmbeddingMatrix = np.array(testSampleEmbeddingMatrix)
            testSampleEmbeddingMatrix = testSampleEmbeddingMatrix.T  # Embedding matrix has vectors in columns instead of rows
            Path(testSampleEmbeddingsFilepath).parent.mkdir(parents=True, exist_ok=True)
            np.savetxt(testSampleEmbeddingsFilepath, testSampleEmbeddingMatrix)
            self.testSampleEmbeddingMatrix = testSampleEmbeddingMatrix
        else:
            self.testSampleEmbeddingMatrix = np.loadtxt(testSampleEmbeddingsFilepath)

        logging.info("Calculating test image product table...")
        testIptFilepath = FilepathUtils.get_sample_test_ipm_filepath(sampleDir, testName)
        if not os.path.isfile(testIptFilepath):
            self.testImageProductTable = calculate_image_product_matrix(self.testImages, get_image_product(
                sampleEstimator.imageProductType))
            Path(testIptFilepath).parent.mkdir(parents=True, exist_ok=True)
            np.savetxt(testIptFilepath, self.testImageProductTable)
        else:
            self.testImageProductTable = np.loadtxt(testIptFilepath)

        logging.info("Calculating plotting data...")
        testSamplePlottingDataFilepath = FilepathUtils.get_sample_test_plotting_data_filepath(sampleDir, testName)
        if not os.path.isfile(testSamplePlottingDataFilepath):
            Path(testSamplePlottingDataFilepath).parent.mkdir(parents=True, exist_ok=True)
            with open(testSamplePlottingDataFilepath, 'wb') as f:
                self.plottingData = get_sample_plotting_data(imageProductMatrix=self.testImageProductTable, embeddingMatrix=self.testSampleEmbeddingMatrix, imagesFilepath=testImagesFilepath)
                pickle.dump(self.plottingData, f)
        else:
            with open(testSamplePlottingDataFilepath, 'rb') as f:
                self.plottingData = pickle.load(f)

def investigate_sample_tester(sampleTester: SampleTester, numSample: int, plotTitle: str):
    plottingData = sampleTester.plottingData
    kNeighFig, axList = plt.subplots(numSample + 1, 2)
    kNeighFig.suptitle("Norm k neighbour plot for " + plotTitle)
    if numSample != 0:
        imgArr = [row[0] for row in axList[:-1]]
        kNeighArr = [row[1] for row in axList[:-1]]
    else:
        imgArr = []
        kNeighArr = []
    aveAx = axList[-1][1]
    # Set the bottom right subplot to be empty
    axList[-1][0].set_axis_off()
    GraphEstimates.plot_swept_k_neighbours(axArr=kNeighArr, imageAxArr=imgArr, aveAx=aveAx,
                                           kNormNeighbourScores=plottingData.kNormNeighbourScore,
                                           aveNormKNeighbourScores=plottingData.aveNormKNeighbourScore,
                                           imagesFilepath=plottingData.imagesFilepath, nImageSample=numSample)

def investigate_pencorr_rank_constraint(*, imageType: str, filters=None, imageProductType: str, startingConstr: int,
                                        endingConstr: int, specifiedKArr=None):
    """
    :param specifiedKArr: value of k for the k neighbour score
    :param imageType:
    :param filters:
    :param imageProductType:
    :param startingConstr: Starting lowest rank constraint to start the sweep inclusive
    :param endingConstr: Final largest rank constraint to end the sweep inclusive
    :return: Uses the penncorr method to generate embeddings for different rank constraints
    Makes a graph of the average neighbour score against rank_constraint and
    average frobenius distance against rank_constraint
    Remember to use plt.show() to display plots

    Aims to answer the question: How does the rank constraint affect the error of the embeddings generated by penncorr?
    """
    if startingConstr >= endingConstr:
        raise ValueError("Starting rank constraint must be lower than ending constraint")
    if specifiedKArr is None:
        specifiedKArr = [5]
    allAveNeighArr = []
    aveFrobDistanceArr = []
    rankConstraints = range(startingConstr, endingConstr + 1)
    for rank in rankConstraints:
        logging.info("Investigating rank " + str(rank) + "/" + str(endingConstr))
        embType = "pencorr_" + str(rank)
        plottingData = data_processing.VecRep.load_BF_plotting_data(imageType=imageType, filters=filters,
                                                                    imageProductType=imageProductType,
                                                                    embeddingType=embType)
        aveNeighArr = []
        for k in specifiedKArr:
            aveNeighArr.append(get_specified_ave_k_neighbour_score(plottingData.aveNormKNeighbourScore, k))
        allAveNeighArr.append(aveNeighArr)
        aveFrobDistanceArr.append(plottingData.aveFrobDistance)


    rankFig, axArr = plt.subplots(1, len(specifiedKArr) + 1)
    frobAx = axArr[-1]
    neighAx = axArr[:-1]
    GraphEstimates.plot_error_against_rank_constraint(frobAx, neighAx, rankConstraints, aveFrobDistanceArr, allAveNeighArr,
                                                      specifiedKArr)

if __name__ == '__main__':
    imageType = "5bin50max_ones"
    filters = ["one_island", "unique"]
    imageProductType = "ncc"
    embeddingType = "pencorr_20"
    overwrite = {"filter": False, "im_prod": False, "estimate": False, 'plot': False, 'sampling': False}
    sampleSize = 400
    testSize = 50
    sampleName = SamplingMethod.get_unique_sample_name(sampleSize)
    testName = SamplingMethod.get_unique_sample_test_name(testSize)
    randSample = get_random_image_sample(imageType=imageType, filters=filters, nSamples=sampleSize+testSize)
    imageSample = randSample[:sampleSize]
    testSample = randSample[sampleSize:]
    sampleEstimator = SampleEstimator(sampleName=sampleName, imageType=imageType,
                                      filters=filters, embeddingType=embeddingType, imageProductType=imageProductType,
                                      overwrite=overwrite,imageSamplesInput=imageSample)
    sampleTest = SampleTester(testImages=testSample, sampleEstimator=sampleEstimator,
                              sampleDir=sampleEstimator.sampleDirectory, testName=testName)
    investigate_sample_tester(sampleTest, 5, "test")
    plt.show()

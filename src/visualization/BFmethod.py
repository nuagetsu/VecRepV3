import logging
import math
import random
import sys

import matplotlib.pyplot as plt

import visualization.Metrics as metrics
from src.data_processing.BruteForceEstimator import BruteForceEstimator
from src.data_processing.TestableEstimator import TestableEstimator
from visualization import GraphEstimates

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def investigate_k(estimator: TestableEstimator, kArr=None, numK=16):
    """
    :param kArr: A specified array of k to create the histograms for. If none, then histograms will be created for all
    k from 1 to numK
    :param numK: Number of K swept before ending the sweep. inclusive
    :param estimator: Testable estimator to investigate
    :return: Creates histogram of k distribution
    Remember to use plt.show() to display plots
    Aims to answer the question: What is the best value to choose for K?
    """
    if kArr is None:
        kArr = [i for i in range(1, numK + 1)]

    # Find the size of the plotting area to fit in all the histograms
    sideLen = math.ceil(math.sqrt(len(kArr)))

    # Creating the list of axes to plot
    fig, axArr = plt.subplots(sideLen, sideLen)
    flat_list = []
    for sub_list in axArr:
        for ele in sub_list:
            flat_list.append(ele)

    # Making the plots for all the histograms
    for kIndex in range(len(kArr)):
        kNeighArr = []
        k = kArr[kIndex]
        ax = flat_list[kIndex]
        for imgIndex in range(len(estimator.matrixG)):
            imgProducts = estimator.matrixG[imgIndex]
            embDotProducts = estimator.matrixGprime[imgIndex]
            kNeighArr.append(metrics.get_k_neighbour_score(imgProducts, embDotProducts, k))
        GraphEstimates.plot_k_histogram(ax, kNeighArr, k)

    fig.suptitle("K neighbour score histograms for " + estimator.to_string())


def investigate_estimator(estimator: TestableEstimator, numK=16, plottedImagesIndex=None, numSample=2):
    """
    :param numK: Values of k to plot for the k neighbour graphs. k will sweep from 1 to numK
    :param estimator: Testable Estimator to investigate
    :param plottedImagesIndex: Index of images you want to plot the k neighbours plot for
    :param numSample: Number of images to plot in the k neighbour plot
    :return: Makes an eigenvalue graph
     swept k neighbours score graph for the mean K val and for a number of images
     and displays the frobenius distance for the embeddings
    Remember to use plt.show() to display plots

    Aims to answer the question: What is the error in using the selected method for generating embeddings?
    """
    # Comparing the largest and the most negative eigenvalues
    eigenFig, (ax1, ax2, ax3) = plt.subplots(3)
    eigenFig.suptitle("Eigenvalue plot and stats of " + estimator.to_string())
    GraphEstimates.plot_eigenvalues(ax1, ax2, estimator.initialEigenvalues, estimator.finalEigenvalues)

    # Displaying stats
    GraphEstimates.plot_key_stats_text(ax3, estimator.frobDistance, estimator.aveFrobDistance, estimator.maxDifference)

    # Making plot for the images and their k neighbour scores
    if plottedImagesIndex is None:
        plottedImagesIndex = random.sample(range(1, len(estimator.imageSet)), numSample)
    else:
        numSample = len(plottedImagesIndex)

    # Creating the subplots for the images and their neighbour plots
    imgFig, axArr = plt.subplots(numSample + 1, 2)
    imgFig.suptitle("K neighbour plot of " + estimator.to_string())

    # Creating the values of K to plot
    kArr = list(range(1, numK + 1))
    count = 0
    # Plotting the image and neighbour plot for each image
    for imageIndex in plottedImagesIndex:
        image = estimator.imageSet[imageIndex]
        imageTitle = "Image " + str(imageIndex)
        imgAx = axArr[count][0]
        neighAx = axArr[count][1]
        kNeighArr = []
        for k in kArr:
            imageProducts = estimator.matrixG[imageIndex]
            dotProducts = estimator.matrixGprime[imageIndex]
            kNeighArr.append(metrics.get_normed_k_neighbour_score(imageProducts, dotProducts, k))
        GraphEstimates.plot_single_image_k_neighbours(imgAx, neighAx, image, imageTitle, kArr, kNeighArr)
        count += 1
    # Plotting the average k score
    aveKNeighArr = []
    for k in kArr:
        aveKNeighArr.append(metrics.get_mean_normed_k_neighbour_score(estimator.matrixG, estimator.matrixGprime, k))

    aveAx = axArr[-1][1]
    GraphEstimates.plot_ave_k_neighbours(aveAx, aveKNeighArr, kArr)

    # Set the bottom right subplot to be empty
    axArr[-1][0].set_axis_off()


def investigate_BF_rank_constraint(*, imageType: str, filters=None, imageProductType: str, startingConstr: int,
                                   endingConstr: int, interval=1, specifiedKArr=None, plotFrob=True):
    """
    :param specifiedKArr: value of k for the k neighbour score graph
    :param imageType:
    :param filters:
    :param imageProductType:
    :param startingConstr: Starting lowest rank constraint to start the sweep inclusive
    :param endingConstr: Final largest rank constraint to end the sweep inclusive
    :param interval: Interval for the rank sweep
    :param plotFrob: If true, also plots the frob error against rank
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

    # A list of k nieghbour plotting data, for each of the k in specified K array
    allAveNeighArr = [[] for i in specifiedKArr]
    aveFrobDistanceArr = []
    rankConstraints = list(range(startingConstr, endingConstr + 1, interval))

    # For each rank constraint, create a BF estimator and get its results
    for rank in rankConstraints:
        logging.info("Investigating rank " + str(rank) + "/" + str(endingConstr))
        embType = "pencorr_" + str(rank)
        bfEstimator = BruteForceEstimator(imageType=imageType, filters=filters,
                                          imageProductType=imageProductType,
                                          embeddingType=embType)

        # For each k to be investigated, append the respective k neighbour score
        for i in range(len(specifiedKArr)):
            k = specifiedKArr[i]
            allAveNeighArr[i].append(metrics.get_mean_normed_k_neighbour_score(bfEstimator.matrixG,
                                                                               bfEstimator.matrixGprime, k))
        aveFrobDistanceArr.append(bfEstimator.aveFrobDistance)

    # Plot an additional graph if plotFrob
    if plotFrob:
        rankFig, axArr = plt.subplots(1, len(specifiedKArr) + 1)
        frobAx = axArr[-1]
        neighAx = axArr[:-1]
        GraphEstimates.plot_frob_error_against_rank_constraint(frobAx, rankConstraints, aveFrobDistanceArr)
    else:
        rankFig, neighAx = plt.subplots(1, len(specifiedKArr))
    GraphEstimates.plot_error_against_rank_constraint(neighAx, rankConstraints, allAveNeighArr, specifiedKArr)

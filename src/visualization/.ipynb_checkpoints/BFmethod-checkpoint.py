import logging
import math
import os.path
import random
import sys
from pathlib import Path
from line_profiler import profile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.helpers.FilepathUtils import get_image_set_filepath, get_set_size_df_filepath
from src.data_processing.BruteForceEstimator import BruteForceEstimator
from src.data_processing.TestableEstimator import TestableEstimator
from src.data_processing.Utilities import get_image_set_size
from src.visualization import GraphEstimates
import src.visualization.Metrics as metrics

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

    # Set the bottom left subplot to be empty
    axArr[-1][0].set_axis_off()


def investigate_BF_rank_constraint(*, imageType: str, filters=None, imageProductType: str, startingConstr: int,
                                   endingConstr: int, interval=1, specifiedKArr=None, plotFrob=True, weight=None,
                                   progressive=False):
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
    if progressive:
        rankConstraints = metrics.get_progressive_range(startingConstr, endingConstr + 1, interval)

    # For each rank constraint, create a BF estimator and get its results
    for rank in rankConstraints:
        logging.info("Investigating rank " + str(rank) + "/" + str(endingConstr))
        embType = "pencorr_" + str(rank)
        bfEstimator = BruteForceEstimator(imageType=imageType, filters=filters,
                                          imageProductType=imageProductType,
                                          embeddingType=embType, weightType=weight)

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
    if type(neighAx) is not list:
        neighAx = [neighAx]
    GraphEstimates.plot_error_against_rank_constraint(neighAx, rankConstraints, allAveNeighArr, specifiedKArr)


def investigate_BF_weight_power(*, imageType: str, filters=None, imageProductTypes: list,
                                startingConstr: int, endingConstr: int, interval=1,
                                specifiedKArr=None, plotFrob=False, rank=192):
    """
    Investigates the effect of increasing powers of G being used as a weighting matrix.
    :param imageType: Image set type to use.
    :param filters: Filters to apply.
    :param imageProductTypes: Image product types to plot for.
    :param startingConstr: Starting power of NCC score to use.
    :param endingConstr: Ending power of NCC score to use.
    :param interval: Interval of increase.
    :param specifiedKArr: Specified values of k to use for k-score
    :param plotFrob: Whether to plot Frobenius Erro (Broken but feature can be removed).
    :param rank: Rank constraint to apply.
    :return: Graph of k-score achieved for a specified rank constraint when weighted pencorr is applied.
    """
    if startingConstr >= endingConstr:
        raise ValueError("Starting rank constraint must be lower than ending constraint")
    if specifiedKArr is None:
        specifiedKArr = [5]

    # A list of k neighbour plotting data, for each of the k in specified K array
    allAveNeighArr = [[[] for imageProductType in imageProductTypes] for k in specifiedKArr]
    weightConstraints = np.arange(startingConstr, endingConstr + interval, interval).tolist()

    for i in range(len(weightConstraints)):
        weight = weightConstraints[i]
        logging.info("Investigating weight " + str(weight) + "/" + str(endingConstr))
        for imageProductTypeIndex in range(len(imageProductTypes)):
            imageProductType = imageProductTypes[imageProductTypeIndex]

            embType = "pencorr_" + str(rank)
            if weight != 0:
                weightType = "ncc_factor_" + str(weight)
            else:
                weightType = None
            log_string = "Investigating image product " + imageProductType

            logging.info(log_string + " for weight " + str(weight) + "/" + str(endingConstr))
            bfEstimator = BruteForceEstimator(imageType=imageType, filters=filters, imageProductType=imageProductType,
                                              embeddingType=embType, weightType=weightType)
            # For each k to be investigated, append the respective k neighbour score
            for kIndex in range(len(specifiedKArr)):
                k = specifiedKArr[kIndex]
                allAveNeighArr[kIndex][imageProductTypeIndex].append(
                    metrics.get_mean_normed_k_neighbour_score(bfEstimator.matrixG, bfEstimator.matrixGprime, k))
    rankFig, neighAx = plt.subplots(1, len(specifiedKArr))
    if type(neighAx) is not np.ndarray:
        neighAx = [neighAx]
    GraphEstimates.plot_ave_k_neighbours_for_weights_in_one(neighAx, weightConstraints, allAveNeighArr,
                                                            specifiedKArr, imageProductTypes,
                                                            imageSet=imageType)


def investigate_BF_rank_constraint_for_image_types(*, imageType: str, filters=None, imageProductTypes: list,
                                                   startingConstr: int, endingConstr: int, interval=1,
                                                   specifiedKArr=None, weights=None, progressive=False,
                                                   embeddings=None):
    """
    :param imageType: Image set to be tested
    :param filters: Filters to produce the image set
    :param imageProductTypes: Image product types to plot
    :param startingConstr: Starting lowest rank constraint to start the sweep inclusive
    :param endingConstr: Final largest rank constraint to end the sweep inclusive
    :param interval: Interval for the rank sweep
    :param specifiedKArr: value of k for the k neighbour score graph
    :param plotFrob: If true, also plots the frob error against rank
    :param weight: Weight matrix to be used in pencorr method
    :return: Uses the penncorr method to generate embeddings for different rank constraints and image product types
    Makes a graph of the average neighbour score against rank_constraint and
    average frobenius distance against rank_constraint
    Remember to use plt.show() to display plots
    """
    if startingConstr >= endingConstr:
        raise ValueError("Starting rank constraint must be lower than ending constraint")
    if specifiedKArr is None:
        specifiedKArr = [5]
    if weights is None:
        weights = ["" for image_product_type in imageProductTypes]
    if embeddings is None:
        embeddings = ["pencorr" for image_product_type in imageProductTypes]

    # A list of k neighbour plotting data, for each of the k in specified K array
    allAveNeighArr = [[[] for imageProductType in imageProductTypes] for k in specifiedKArr]
    rankConstraints = list(range(startingConstr, endingConstr + 1, interval))
    if progressive:
        rankConstraints = metrics.get_progressive_range(startingConstr, endingConstr + 1, interval)

    for i in range(len(rankConstraints)):
        rank = rankConstraints[i]
        logging.info("Investigating rank " + str(rank) + "/" + str(endingConstr))
        for imageProductTypeIndex in range(len(imageProductTypes)):
            imageProductType = imageProductTypes[imageProductTypeIndex]
            weight = weights[imageProductTypeIndex]
            embType = embeddings[imageProductTypeIndex] + "_" + str(rank)
            log_string = "Investigating image product " + imageProductType

            if weight != "":
                log_string += " with weight of " + weight
            log_string += " and embedding type " + embeddings[imageProductTypeIndex]
            logging.info(log_string + " for rank " + str(rank) + "/" + str(endingConstr))
            bfEstimator = BruteForceEstimator(imageType=imageType, filters=filters, imageProductType=imageProductType,
                                              embeddingType=embType, weightType=weight)
            # For each k to be investigated, append the respective k neighbour score
            for kIndex in range(len(specifiedKArr)):
                k = specifiedKArr[kIndex]
                allAveNeighArr[kIndex][imageProductTypeIndex].append(
                    metrics.get_mean_normed_k_neighbour_score(bfEstimator.matrixG, bfEstimator.matrixGprime, k))
    rankFig, neighAx = plt.subplots(1, len(specifiedKArr))
    if type(neighAx) is not np.ndarray:
        neighAx = [neighAx]
    GraphEstimates.plot_ave_k_neighbours_for_type_in_one(neighAx, rankConstraints, allAveNeighArr,
                                                         specifiedKArr, imageProductTypes,
                                                         weights, embeddings, imageSet=imageType)


def investigate_image_product_type(*, imageType: str, filters=None, imageProductTypeArr=None, embType: str,
                                   numK=16, plotFrob=True, overwrite=None, weight=None):
    """
    :param imageType: The image set to investigate
    :param filters: Filters to apply to the image set
    :param imageProductTypeArr: Image product types to investigate
    :param plotFrob: If true, also displays the frobenius error
    :param weight: Weight to use in embedding process
    :return: Plots a graph of the relative positioning score against the number of neighbours analysed for each image
        product type in the image product type array.
    """
    if imageProductTypeArr is None:
        imageProductTypeArr = ["ncc"]

    imgFig, axArr = plt.subplots(len(imageProductTypeArr), 1 + plotFrob)
    imgFig.suptitle("Relative Positioning Score of image product types on " + imageType + ", " + embType)

    count = 0
    for imageProductType in imageProductTypeArr:
        logging.info("Investigating image product " + imageProductType)
        bfEstimator = BruteForceEstimator(imageType=imageType, filters=filters, imageProductType=imageProductType,
                                          embeddingType=embType, overwrite=overwrite, weightType=weight)
        kArr = list(range(1, numK + 1))
        aveKNeighArr = []
        for k in kArr:
            aveKNeighArr.append(metrics.get_mean_normed_k_neighbour_score(bfEstimator.matrixG,
                                                                          bfEstimator.matrixGprime, k))
        plotPoint = axArr[count]
        if plotFrob:
            GraphEstimates.plot_key_stats_text(axArr[count][1], bfEstimator.frobDistance, bfEstimator.aveFrobDistance,
                                               bfEstimator.maxDifference)
            plotPoint = plotPoint[0]
        GraphEstimates.plot_ave_k_neighbours_for_type(plotPoint, aveKNeighArr, kArr, imageProductType)
        count += 1

@profile
def investigate_plateau_rank_for_set_sizes(*, image_types: list, filters=None, image_product_types: list, embeddings: list,
                                           weights: list, k=5, prox=3, overwrite=None):
    """
    Investigates and plots plateau rank for image sets of increasing set sizes.
    :param image_types: Image sets to use
    :param filters: Filters to apply
    :param image_product_types: Image product types to be used.
    :param embeddings: Embedding methods to use.
    :param weights: Weights to be used if applicable.
    :param k: k values for which to find k-score.
    :param prox: Proximity of binary search for plateau rank.
    :param overwrite: Whether to overwrite data.
    :return: Plots a graph of plateau rank against image set sizes for image sets of increasing size, ideally
    with the same image sizes.
    """
    data = {"Image Set": image_types, "Image Size": [], "Image Products": image_product_types, "Embeddings": embeddings, "Weights": weights,
            "K_scores": [], "Set Size": [], "Non_zero": [], "Plateau Rank": []}
    set_groups = {}
    for index, image_type in enumerate(image_types):
        logging.info("Investigating " + image_type)

        # Setting variables
        image_product = image_product_types[index]
        weight = weights[index]
        embedding = embeddings[index]
        image_set_filepath = get_image_set_filepath(image_type, filters)
        image_set_size = get_image_set_size(image_type, filters, image_set_filepath)
        data["Set Size"].append(image_set_size)
        logging.info("Image set size is " + str(image_set_size))
        image_size = 0

        # Loop variables
        high = image_set_size
        low = 0
        selected_rank = high
        max_k_score = 2
        iterations = 0
        same_rank = 0
        score_change = False
        max_score_rank = []

        # Begin binary search. Search ends when high estimate is within "prox" of low estimate
        while high - low > prox:
            logging.info("Starting iteration " + str(iterations + 1))
            selected_embedding = embedding + "_" + str(selected_rank)
            bfEstimator = BruteForceEstimator(imageType=image_type, filters=filters, imageProductType=image_product,
                                              embeddingType=selected_embedding, overwrite=overwrite, weightType=weight)
            k_score = metrics.get_mean_normed_k_neighbour_score(bfEstimator.matrixG, bfEstimator.matrixGprime, k)
            if not score_change:
                # k_score is the same as previous tested rank constraint
                if iterations == 0:
                    # First iteration, no rank constraint placed
                    max_k_score = k_score   # k_score value where plateau occurs
                    data["K_scores"].append(max_k_score)
                    nonzero = np.count_nonzero(np.array([np.max(b) - np.min(b) for b in bfEstimator.matrixA]))
                    data["Non_zero"].append(nonzero)    # Number of nonzero eigenvalues after pencorr acts as upper
                    high = nonzero                      # bound for plateau rank
                    low = nonzero // 2
                    image_size = len(bfEstimator.imageSet[0])
                    data["Image Size"].append(image_size)
                elif k_score == max_k_score:
                    # Not first iteration, k_score has yet to change. Continue lowering rank constraint.
                    high = low
                    low = high // 2
                else:
                    # Not first iteration, k_score has changed. Begin looking for plateau rank.
                    score_change = True
                    low = ((high - low) // 2) + low
            elif k_score != max_k_score:
                # Before plateau area. Raise rank constraint.
                low = ((high - low) // 2) + low
                max_score_rank = []
                same_rank = 0
            elif same_rank == 2:
                # Successively within plateau area. Break loop
                high = max_score_rank[0]
                iterations += 1
                logging.info("Finishing iteration" + str(iterations))
                break
            else:
                # Within plateau area. Raise rank constraint slowly in case plateau rank not yet reached.
                max_score_rank.append(low)
                diff = (high - low) // 4
                low += diff
                same_rank += 1
            logging.info("k score is " + str(k_score))

            # Test next iteration at low estimate
            selected_rank = low
            iterations += 1
            logging.info("Finishing iteration " + str(iterations))
            logging.info("Next Rank " + str(low))

        # Once loop ends, save high estimate plateau rank for currently tested image set
        logging.info("Plateau rank " + str(high))
        data["Plateau Rank"].append(high)

        label = image_product + ", " + embedding + ", size " + str(image_size) + " by " + str(image_size)
        if label not in set_groups:
            set_groups[label] = {"plateau ranks": [], "Image Set Size": []}
        set_groups[label]["plateau ranks"].append(high)
        set_groups[label]["Image Set Size"].append(image_set_size)

    df = pd.DataFrame(data)
    filepath = get_set_size_df_filepath(image_types)
    if not os.path.isfile(filepath):
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filepath, index=False)
    print(df)

    GraphEstimates.plot_plateau_ranks_categorised(set_groups, tag="Image Set Size")


def investigate_goal_rank_for_set_sizes(*, image_types: list, filters=None, image_product_types: list, embeddings: list,
                                        weights: list, k=5, prox=3, overwrite=None):
    """
    Investigates and plots a graph of goal rank against image set sizes for several image sets.
    WIP. TODO Fix this
    :param image_types: Image sets to plot for.
    :param filters: Filters to apply.
    :param image_product_types: Image products to be used for each image set.
    :param embeddings: Embedding methods to be used fpr each image set.
    :param weights: Weighting types to be used.
    :param k: values of k for which k-score is calculated
    :param prox: Proximity at which the binary search is terminated.
    :param overwrite: Whether to overwrite data.
    :return: Plots a graph of goal rank against image set sizes.
    """
    data = {"Image Set": image_types, "Image Size": [], "Image Products": image_product_types, "Embeddings": embeddings,
            "Weights": weights,
            "K_scores": [], "Set Size": [], "Non_zero": [], "Goal Rank": []}
    set_groups = {}
    goal_k_score = 2
    for index, image_type in enumerate(image_types):
        logging.info("Investigating " + image_type)

        # Setting variables
        image_product = image_product_types[index]
        weight = weights[index]
        embedding = embeddings[index]
        image_set_filepath = get_image_set_filepath(image_type, filters)
        image_set_size = get_image_set_size(image_type, filters, image_set_filepath)
        data["Set Size"].append(image_set_size)
        logging.info("Image set size is " + str(image_set_size))
        image_size = 0

        # Loop variables
        high = image_set_size
        low = 0
        selected_rank = high
        iterations = 0
        same_rank = 0
        score_change = False
        max_score_rank = []

        if index == 0:
            # Begin binary search. Search ends when high estimate is within "prox" of low estimate
            # For the first set, do plateau rank search and look for when max k_score is achieved
            while high - low > prox:
                logging.info("Starting iteration " + str(iterations + 1))
                selected_embedding = embedding + "_" + str(selected_rank)
                bfEstimator = BruteForceEstimator(imageType=image_type, filters=filters, imageProductType=image_product,
                                                  embeddingType=selected_embedding, overwrite=overwrite, weightType=weight)
                k_score = metrics.get_mean_normed_k_neighbour_score(bfEstimator.matrixG, bfEstimator.matrixGprime, k)
                if not score_change:
                    # k_score is the same as previous tested rank constraint
                    if iterations == 0:
                        # First iteration, no rank constraint placed
                        goal_k_score = k_score  # k_score value where plateau occurs
                        data["K_scores"].append(goal_k_score)
                        nonzero = np.count_nonzero(np.array([np.max(b) - np.min(b) for b in bfEstimator.matrixA]))
                        data["Non_zero"].append(nonzero)  # Number of nonzero eigenvalues after pencorr acts as upper
                        high = nonzero  # bound for plateau rank
                        low = nonzero // 2
                        image_size = len(bfEstimator.imageSet[0])
                        data["Image Size"].append(image_size)
                    elif k_score == goal_k_score:
                        # Not first iteration, k_score has yet to change. Continue lowering rank constraint.
                        high = low
                        low = high // 2
                    else:
                        # Not first iteration, k_score has changed. Begin looking for plateau rank.
                        score_change = True
                        low = ((high - low) // 2) + low
                elif k_score != goal_k_score:
                    # Before plateau area. Raise rank constraint.
                    low = ((high - low) // 2) + low
                    max_score_rank = []
                    same_rank = 0
                elif same_rank == 2:
                    # Successively within plateau area. Break loop
                    high = max_score_rank[0]
                    iterations += 1
                    logging.info("Finishing iteration" + str(iterations))
                    break
                else:
                    # Within plateau area. Raise rank constraint slowly in case plateau rank not yet reached.
                    max_score_rank.append(low)
                    diff = (high - low) // 4
                    low += diff
                    same_rank += 1
                logging.info("k score is " + str(k_score))

                # Test next iteration at low estimate
                selected_rank = low
                iterations += 1
                logging.info("Finishing iteration " + str(iterations))
                logging.info("Next Rank " + str(low))
        else:
            # For subsequent sets, we want to find the rank where we hit the goal k_score
            while high - low > prox:
                logging.info("Starting iteration " + str(iterations + 1))
                selected_embedding = embedding + "_" + str(selected_rank)
                bfEstimator = BruteForceEstimator(imageType=image_type, filters=filters, imageProductType=image_product,
                                                  embeddingType=selected_embedding, overwrite=overwrite, weightType=weight)
                k_score = metrics.get_mean_normed_k_neighbour_score(bfEstimator.matrixG, bfEstimator.matrixGprime, k)
                if not score_change:
                    # k_score is the same as previous tested rank constraint
                    if iterations == 0:
                        # First iteration, no rank constraint placed
                        max_k_score = k_score  # k_score value where plateau occurs
                        data["K_scores"].append(max_k_score)
                        nonzero = np.count_nonzero(np.array([np.max(b) - np.min(b) for b in bfEstimator.matrixA]))
                        data["Non_zero"].append(nonzero)  # Number of nonzero eigenvalues after pencorr acts as upper
                        high = nonzero  # bound for plateau rank
                        low = nonzero // 2
                        image_size = len(bfEstimator.imageSet[0])
                        data["Image Size"].append(image_size)
                        if max_k_score < goal_k_score:
                            logging.info("Goal k-score cannot be achieved! Skipping this image set.")
                            break
                    elif k_score > goal_k_score:
                        # Not first iteration, k_score is still higher. Continue lowering rank constraint.
                        high = low
                        low = high // 2
                    elif k_score == goal_k_score:
                        logging.info("Goal k-score achieved.")
                    else:
                        # Not first iteration, k_score has is lower. Begin looking for goal rank.
                        score_change = True
                        low = ((high - low) // 2) + low
                elif k_score < goal_k_score:
                    # Before goal k-score
                    low = ((high - low) // 2) + low
                elif k_score == goal_k_score:
                    # Goal k-score found
                    high = low
                    iterations += 1
                    logging.info("Finishing iteration" + str(iterations))
                    break
                else:
                    # Above goal k-score
                    high = low
                    low = 3 * (low // 4)
                logging.info("k score is " + str(k_score))

                # Test next iteration at low estimate
                selected_rank = low
                iterations += 1
                logging.info("Finishing iteration " + str(iterations))
                logging.info("Next Rank " + str(low))

            # Once loop ends, save high estimate plateau rank for currently tested image set
            logging.info("Goal rank " + str(high))
            data["Goal Rank"].append((high + low) // 2)

        label = image_product + ", " + embedding + ", size " + str(image_size) + " by " + str(image_size)
        if label not in set_groups:
            set_groups[label] = {"goal ranks": [], "Image Set Size": []}
        set_groups[label]["goal ranks"].append(high)
        set_groups[label]["Image Set Size"].append(image_set_size)

    df = pd.DataFrame(data)
    print(df)

    GraphEstimates.plot_goal_ranks_categorised(set_groups, goal_k_score, tag="Image Set Size")


def investigate_plateau_rank_for_image_sizes(*, image_types: list, filters=None, image_product_types: list, embeddings: list,
                             weights: list, k=5, prox=3, overwrite=None):
    """
    Investigates and plots a graph for plateau rank against several image sets of different sizes, preferably keeping
    image set size constant.
    :param image_types: Image types to plot for.
    :param filters: Filters to be applied.
    :param image_product_types: Image product types to use for each image set.
    :param embeddings: Embedding methods to use for each image set.
    :param weights: Weight types to use for each image set.
    :param k: k values to use for calculating k-scores.
    :param prox: Proximity at which to terminate the binary search.
    :param overwrite: Whether to generate new data even if saved data is found.
    :return: Plots a graph of plateau rank against image set size. Also generates a dataframe to refer to.
    """
    data = {"Image Set": image_types, "Image Size": [], "Image Products": image_product_types, "Embeddings": embeddings,
            "Weights": weights,
            "K_scores": [], "Set Size": [], "Non_zero": [], "Plateau Rank": []}
    set_groups = {}
    for index, image_type in enumerate(image_types):
        logging.info("Investigating " + image_type)

        # Setting variables
        image_product = image_product_types[index]
        weight = weights[index]
        embedding = embeddings[index]
        image_set_filepath = get_image_set_filepath(image_type, filters)
        image_set_size = get_image_set_size(image_type, filters, image_set_filepath)
        data["Set Size"].append(image_set_size)
        logging.info("Image set size is " + str(image_set_size))
        image_size = 0

        # Loop variables
        high = image_set_size
        low = 0
        selected_rank = high
        max_k_score = 2
        iterations = 0
        same_rank = 0
        score_change = False
        max_score_rank = []

        # Begin binary search. Search ends when high estimate is within "prox" of low estimate
        while high - low > prox:
            logging.info("Starting iteration " + str(iterations + 1))
            selected_embedding = embedding + "_" + str(selected_rank)
            bfEstimator = BruteForceEstimator(imageType=image_type, filters=filters, imageProductType=image_product,
                                              embeddingType=selected_embedding, overwrite=overwrite, weightType=weight)
            k_score = metrics.get_mean_normed_k_neighbour_score(bfEstimator.matrixG, bfEstimator.matrixGprime, k)
            if not score_change:
                # k_score is the same as previous tested rank constraint
                if iterations == 0:
                    # First iteration, no rank constraint placed
                    max_k_score = k_score  # k_score value where plateau occurs
                    data["K_scores"].append(max_k_score)
                    nonzero = np.count_nonzero(np.array([np.max(b) - np.min(b) for b in bfEstimator.matrixA]))
                    data["Non_zero"].append(nonzero)  # Number of nonzero eigenvalues after pencorr acts as upper
                    high = nonzero  # bound for plateau rank
                    low = nonzero // 2
                    image_size = len(bfEstimator.imageSet[0])
                    data["Image Size"].append(image_size)
                elif k_score == max_k_score:
                    # Not first iteration, k_score has yet to change. Continue lowering rank constraint.
                    high = low
                    low = high // 2
                else:
                    # Not first iteration, k_score has changed. Begin looking for plateau rank.
                    score_change = True
                    low = ((high - low) // 2) + low
            elif k_score != max_k_score:
                # Before plateau area. Raise rank constraint.
                low = ((high - low) // 2) + low
                max_score_rank = []
                same_rank = 0
            elif same_rank == 2:
                # Successively within plateau area. Break loop
                high = max_score_rank[0]
                iterations += 1
                logging.info("Finishing iteration" + str(iterations))
                break
            else:
                # Within plateau area. Raise rank constraint slowly in case plateau rank not yet reached.
                max_score_rank.append(low)
                diff = (high - low) // 4
                low += diff
                same_rank += 1
            logging.info("k score is " + str(k_score))

            # Test next iteration at low estimate
            selected_rank = low
            iterations += 1
            logging.info("Finishing iteration " + str(iterations))
            logging.info("Next Rank " + str(low))

        # Once loop ends, save high estimate plateau rank for currently tested image set
        logging.info("Plateau rank " + str(high))
        data["Plateau Rank"].append(high)

        label = image_product + ", " + embedding + ", set size " + str(image_set_size)
        if label not in set_groups:
            set_groups[label] = {"plateau ranks": [], "Image Size": []}
        set_groups[label]["plateau ranks"].append(high)
        set_groups[label]["Image Size"].append(image_size)

    df = pd.DataFrame(data)
    print(df)

    GraphEstimates.plot_plateau_ranks_categorised(set_groups, tag="Image Size")

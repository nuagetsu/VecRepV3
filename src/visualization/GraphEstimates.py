import os.path
import random
from statistics import median
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from numpy._typing import NDArray

from data_processing import FilepathUtils
from visualization.Metrics import PlottingData


def plot_eigenvalues(ax1: Axes, ax2: Axes, initialEigenvalues: NDArray, finalEigenvalues: NDArray):
    """
    :param ax1: axes to plot the largest 20%/top 15 eigenvalues
    :param ax2: axes to plot the lowest 20%/bottom 15 eigenvalues
    :param initialEigenvalues: intial eigenvlaues of the image product matrix
    :param finalEigenvalues: final eigenvalues of the embedding matrix dot product
    :return:
    """
    barWidth = 0.4

    numPlot = min(int(len(initialEigenvalues) * 0.2), 15)

    topInitEigen = initialEigenvalues[:numPlot]
    topFinalEigen = finalEigenvalues[:numPlot]
    # Set position of bar on X axis
    br1 = np.arange(numPlot)
    br2 = [x + barWidth for x in br1]
    ax1.set_title("Top " + str(numPlot) + " eigenvalues")
    rects1 = ax1.bar(br1, topInitEigen, color='r', width=barWidth, label='IT')
    rects2 = ax1.bar(br2, topFinalEigen, color='g', width=barWidth, label='ECE')
    ax1.legend((rects1[0], rects2[0]), ('Initial Eigenvalues', 'Final eigenvalues'))

    bottomInitEigen = initialEigenvalues[-numPlot:]
    bottomFinalEigen = finalEigenvalues[-numPlot:]

    ax2.set_title("Bottom " + str(numPlot) + " eigenvalues")
    rects1 = ax2.bar(br1, bottomInitEigen, color='r', width=barWidth, label='IT')
    rects2 = ax2.bar(br2, bottomFinalEigen, color='g', width=barWidth, label='ECE')
    ax2.legend((rects1[0], rects2[0]), ('Initial Eigenvalues', 'Final eigenvalues'))


def plot_swept_ave_k_neighbours(ax, aveKNeighbourScores: List, numPlottedK=None):
    if numPlottedK is None:
        numPlottedK = len(aveKNeighbourScores)
    if numPlottedK > len(aveKNeighbourScores):
        raise ValueError("Choose a lower value for num plotted K")
    idealPlot = range(1, numPlottedK + 1)  # for plotting y=x
    aveX = []
    aveY = []
    for score in aveKNeighbourScores[: numPlottedK]:
        aveX.append(score["kval"])
        aveY.append(score["neighbourScore"])
    ax.plot(idealPlot, [1 for count in range(len(idealPlot))], color='b', linestyle=':', label="Ideal")
    ax.plot(aveX, aveY, color='r', label="Real")
    ax.set_title("Mean neighbour score of all images against number of neighbours analysed")
    ax.set_xlabel("Value of k")
    ax.set_ylabel("Norm K neighbour score")
    ax.set_ylim(0,1.1)
    ax.legend(loc="upper left")


def plot_swept_k_neighbours(*, axArr: List[Axes], imageAxArr: List[Axes], aveAx: Axes, kNormNeighbourScores: List,
                            aveNormKNeighbourScores: List,
                            imagesFilepath: str, nImageSample=3, numPlottedK=None, imageIndexArr=None):
    """
    :param axArr: axes for the k neighbour plots for each image
    :param imageAxArr: axes for the image plot itself
    :param aveAx: axes for the average k neighbour plots
    :param kNormNeighbourScores: list of k neighbour scores from PlottingData
    :param aveNormKNeighbourScores: list of ave k neighbour scores from PlottingData
    :param imagesFilepath: Imagefile path where the images are stored (must be loaded previously)
    :param nImageSample: Number of images to plot their k neighbour score as the value of k changes. 3 by default
    :param numPlottedK: The number of k to sweep. Start the sweep from 1 and ends at numPlottedK - 1. By default plots the max possible number of k stored in plotting data
    :param imageIndexArr: The index of images for which you want to see the k neighbour plot. Images are randomly selected by default
    :return: Plots a graph of norm k neighbour score against the value of K for a sample of images, as well as the mean norm k neighbour score against the value of k
    """
    num_images = len(kNormNeighbourScores[0]["neighbourScore"])
    if nImageSample > num_images:
        raise ValueError("nImageSample is greater than the number of images")
    if len(axArr) != nImageSample:
        raise ValueError("Please input the correct number of axes")
    if len(imageAxArr) != nImageSample:
        raise ValueError("Please input the correct number of images axes")
    if not os.path.exists(imagesFilepath):
        raise FileNotFoundError(imagesFilepath + " does not exist")
    if numPlottedK is None:
        numPlottedK = len(kNormNeighbourScores)
    if numPlottedK > len(kNormNeighbourScores):
        raise ValueError("Choose a lower value for num plotted K")

    if nImageSample != 0:
        # Choose a random sample of images
        if imageIndexArr is None:
            imageIndexArr = random.sample(range(1, num_images), nImageSample)
        elif max(imageIndexArr) > num_images:
            raise ValueError("Invalid image index entered")
        images = np.load(imagesFilepath)
        idealPlot = range(1, numPlottedK + 1)  # for plotting y=1
        for count in range(nImageSample):
            imageNum = imageIndexArr[count]
            ax = axArr[count]
            x = []
            y = []
            for i in range(numPlottedK):
                x.append(kNormNeighbourScores[i]["kval"])
                y.append(kNormNeighbourScores[i]["neighbourScore"][imageNum])
            ax.plot(idealPlot, [1 for count in range(len(idealPlot))], color='b', linestyle=':', label="Ideal")
            ax.plot(x, y, color='r', label="Real")
            ax.set_title("Normed k neighbour score of image " + str(imageNum) + " against number of neighbours analysed")
            ax.set_xlabel("Value of k")
            ax.set_ylabel("Normed K neighbour score")
            ax.set_ylim(0, 1.1)
            ax.legend(loc="upper left")

            imageAx = imageAxArr[count]
            choosenImage = images[imageNum]
            imageAx.set_title("Image " + str(imageNum))
            imageAx.imshow(choosenImage, cmap='Greys', interpolation='nearest')

    plot_swept_ave_k_neighbours(aveAx, aveNormKNeighbourScores, numPlottedK)


def plot_key_stats_text(ax: Axes, plottingData: PlottingData):
    displayText = ("Frobenius norm of difference between imageProductMatrix and A^tA: " + "{:.2f}".format(
        plottingData.frobDistance) + "\n" +
                   "Average Frobenius norm of difference between imageProductMatrix and A^tA: " + "{:.3E}".format(
                plottingData.aveFrobDistance) + "\n" +
                   "Greatest single element difference between imageProductMatrix and A^tA: " + "{:.2f}".format(
                plottingData.maxDiff) + "\n")
    ax.text(0.5, 0.5, displayText, color='black',
            bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=1'), ha='center', va='center')


def plot_comparison_btw_img_prod(ax1Arr: List[Axes], ax2Arr: List[Axes], imageProdType1: str, imageProdType2: str,
                                 plottingData1: PlottingData, plottingData2: PlottingData):
    plot_eigenvalues(ax1Arr[0], ax1Arr[1], plottingData1.initialEigenvalues, plottingData1.finalEigenvalues)
    plot_eigenvalues(ax2Arr[0], ax2Arr[1], plottingData2.initialEigenvalues, plottingData2.finalEigenvalues)
    plot_swept_ave_k_neighbours(ax1Arr[2], plottingData1.aveNormKNeighbourScore)
    plot_swept_ave_k_neighbours(ax2Arr[2], plottingData2.aveNormKNeighbourScore)
    plot_key_stats_text(ax1Arr[3], plottingData1)
    plot_key_stats_text(ax2Arr[3], plottingData2)


def plot_error_against_rank_constraint(frobAx: Axes, neighbourAx: Axes, rankArr: List, frobArr: List, neighArr: List,
                                       specifiedK: int):
    """
    :param frobAx: Axes to plot the frobenius graph
    :param neighbourAx: Axes to plot the neighbour graph
    :param rankArr: array of rank constrain values to plot (x axis for both graphs)
    :param frobArr: Data for the frob graph (y axis)
    :param neighArr: Data for the neighbour graph (y axis)
    :param specifiedK: The k neighbour score used
    :return:
    """
    frobAx.plot(rankArr, frobArr)
    frobAx.set_title("Average frobenius error against rank constraint")
    frobAx.set_xlabel("Rank Constraint")
    frobAx.set_ylabel("Average frobenius error")

    idealPlot = [1 for i in range(len(rankArr))]  # for plotting the max possible score
    neighbourAx.plot(rankArr, idealPlot, color='b', linestyle=':', label="Ideal")
    neighbourAx.plot(rankArr, neighArr, color='r', label="Real")
    neighbourAx.set_title("Mean norm k neighbour score of all images against the rank constraint applied to pencorr (k = " + str(specifiedK) + ")")
    neighbourAx.set_xlabel("Rank Constraint")
    neighbourAx.set_ylabel("K neighbour score (k = " + str(specifiedK) + ")")
    neighbourAx.legend(loc="upper left")

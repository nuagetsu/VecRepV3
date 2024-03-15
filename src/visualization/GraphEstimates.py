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


def plot_ave_k_neighbours(ax, aveKNeighbourScores: List, numPlottedK=None):
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
    ax.plot(idealPlot, idealPlot, color='b', linestyle=':', label="Ideal")
    ax.plot(aveX, aveY, color='r', label="Real")
    ax.set_title("Mean neighbour score of all images against number of neighbours analysed")
    ax.set_xlabel("Value of k")
    ax.set_ylabel("K neighbour score")
    ax.legend(loc="upper left")


def plot_k_neighbours(*, axArr: List[Axes], imageAxArr: List[Axes], aveAx: Axes, kNeighbourScores: List,
                      aveKNeighbourScores: List,
                      imagesFilepath: str, nImageSample=3, numPlottedK=None):
    num_images = len(kNeighbourScores[0]["neighbourScore"])
    if nImageSample > num_images:
        raise ValueError("nImageSample is greater than the number of images")
    if len(axArr) != nImageSample:
        raise ValueError("Please input the correct number of axes")
    if len(imageAxArr) != nImageSample:
        raise ValueError("Please input the correct number of images axes")
    if not os.path.exists(imagesFilepath):
        raise FileNotFoundError(imagesFilepath + " does not exist")
    if numPlottedK is None:
        numPlottedK = len(kNeighbourScores)
    if numPlottedK > len(kNeighbourScores):
        raise ValueError("Choose a lower value for num plotted K")

    # Choose a random sample of images
    random_samples = random.sample(range(1, num_images), nImageSample)
    images = np.load(imagesFilepath)
    idealPlot = range(1, numPlottedK + 1)  # for plotting y=x
    for count in range(nImageSample):
        imageNum = random_samples[count]
        ax = axArr[count]
        x = []
        y = []
        for i in range(numPlottedK):
            x.append(kNeighbourScores[i]["kval"])
            y.append(kNeighbourScores[i]["neighbourScore"][imageNum])
        ax.plot(idealPlot, idealPlot, color='b', linestyle=':', label="Ideal")
        ax.plot(x, y, color='r', label="Real")
        ax.set_title("Neighbour score of image " + str(imageNum) + " against number of neighbours analysed")
        ax.set_xlabel("Value of k")
        ax.set_ylabel("K neighbour score")
        ax.legend(loc="upper left")

        imageAx = imageAxArr[count]
        choosenImage = images[imageNum]
        imageAx.set_title("Image " + str(imageNum))
        imageAx.imshow(choosenImage, cmap='Greys', interpolation='nearest')

    plot_ave_k_neighbours(aveAx, aveKNeighbourScores, numPlottedK)


def plot_key_stats_text(ax: Axes, plottingData: PlottingData):
    displayText = ("Total Frobenius distance between imageProductMatrix and A^tA: " + "{:.2f}".format(
        plottingData.frobDistance) + "\n" +
                   "Average Frobenius distance between imageProductMatrix and A^tA: " + "{:.3E}".format(
                plottingData.aveFrobDistance) + "\n" +
                   "Max Frobenius distance between imageProductMatrix and A^tA: " + "{:.2f}".format(
                plottingData.maxDiff) + "\n")
    ax.text(0.5, 0.5, displayText, color='black',
            bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=1'), ha='center', va='center')


def plot_comparison_btw_img_prod(ax1Arr: List[Axes], ax2Arr: List[Axes], imageProdType1: str, imageProdType2: str,
                                 plottingData1: PlottingData, plottingData2: PlottingData):
    plot_eigenvalues(ax1Arr[0], ax1Arr[1], plottingData1.initialEigenvalues, plottingData1.finalEigenvalues)
    plot_eigenvalues(ax2Arr[0], ax2Arr[1], plottingData2.initialEigenvalues, plottingData2.finalEigenvalues)
    plot_ave_k_neighbours(ax1Arr[2], plottingData1.aveKNeighbourScore)
    plot_ave_k_neighbours(ax2Arr[2], plottingData2.aveKNeighbourScore)
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

    idealPlot = [specifiedK for i in range(len(rankArr))]  # for plotting the max possible score
    neighbourAx.plot(rankArr, idealPlot, color='b', linestyle=':', label="Ideal")
    neighbourAx.plot(rankArr, neighArr, color='r', label="Real")
    neighbourAx.set_title("Mean neighbour score of all images against the rank constraint applied to pencorr")
    neighbourAx.set_xlabel("Rank Constraint")
    neighbourAx.set_ylabel("K neighbour score")
    neighbourAx.legend(loc="upper left")

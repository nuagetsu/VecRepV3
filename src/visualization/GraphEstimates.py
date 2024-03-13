import os.path
import random

from Metrics import PlottingData
import matplotlib.pyplot as plt
import numpy as np
from data_processing import FilepathUtils


def plot_eigenvalues(ax1, ax2, initialEigenvalues, finalEigenvalues):
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

def plot_k_neighbours(axList, imageAxList, kNeighbourScores, imagesFilepath, nImageSample = 3):
    num_images = len(kNeighbourScores[0]["neighbourScore"])
    if nImageSample > num_images:
        raise ValueError("nImageSample is greater than the number of images")
    if len(axList) != nImageSample:
        raise ValueError("Please input the correct number of axes")
    if len(imageAxList) != nImageSample:
        raise ValueError("Please input the correct number of images axes")
    if not os.path.exists(imagesFilepath):
        raise FileNotFoundError(imagesFilepath + " does not exist")

    #Choose a random sample of images
    random_samples = random.sample(range(1, num_images), nImageSample)
    images = np.load(imagesFilepath)
    for count in range(nImageSample):
        imageNum = random_samples[count]
        ax = axList[count]
        x = []
        y = []
        for k in range(len(kNeighbourScores)):
            x.append(kNeighbourScores[k]["kval"])
            y.append(kNeighbourScores[k]["neighbourScore"][imageNum])
        idealPlot = range(1, len(kNeighbourScores) + 1)
        ax.plot(idealPlot, idealPlot, color='b', linestyle=':', label="Ideal")
        ax.plot(x, y, color='r', label="Real")
        ax.set_title("Neighbour score of image " + str(imageNum) + " against number of neighbours analysed")
        ax.set_xlabel("Value of k")
        ax.set_ylabel("K neighbour score")

        imageAx = imageAxList[count]
        choosenImage = images[imageNum]
        imageAx.set_title("Image " + str(imageNum))
        imageAx.imshow(choosenImage, cmap='Greys',  interpolation='nearest')


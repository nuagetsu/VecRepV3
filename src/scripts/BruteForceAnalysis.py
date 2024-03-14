import matplotlib.pyplot as plt

from data_processing import VecRep, FilepathUtils
from visualization import Metrics, GraphEstimates
from visualization.Metrics import PlottingData


def investigate_k(plottingData: PlottingData, numK=None):
    """
    :param numK: Number of K swept before ending the sweep. inclusive
    :param plottingData:
    :return: Creates graphs to visualize how the chosen value of k neighbour score affects the neighbour score for the input plotting data
    Remember to use plt.show() to display plots
    Aims to answer the question: What is the best value to choose for K?
    """

    scoreFig, ([imgAx1, ax1], [imgAx2, ax2], [imgAx3, ax3], [temp, aveAx]) = plt.subplots(4, 2, width_ratios=[1, 3])
    axArr = [ax1, ax2, ax3]
    imgAxArr = [imgAx1, imgAx2, imgAx3]
    temp.set_axis_off()
    GraphEstimates.plot_k_neighbours(axArr=axArr, imageAxArr=imgAxArr, aveAx=aveAx,
                                     kNeighbourScores=plottingData.kNeighbourScores,
                                     imagesFilepath=plottingData.imagesFilepath, numPlottedK=numK)


def investigate_BF_method(plottingData: PlottingData):
    """
    :param plottingData:
    :return: Makes an eigenvalue graph and displays the frobenius distance for the embeddings
    Remember to use plt.show() to display plots

    Aims to answer the question: What is the error in using the selected method for generating embeddings?
    """
    # Comparing the largest and the most negative eigenvalues
    eigenFig, (ax1, ax2) = plt.subplots(2)
    GraphEstimates.plot_eigenvalues(ax1, ax2, plottingData.initialEigenvalues, plottingData.finalEigenvalues)

    # Display key values about the plot
    statsPlot, ax = plt.subplots()
    GraphEstimates.plot_key_stats_text(ax, plottingData)


if __name__ == '__main__':
    imageType = "3bin"
    filters = ["unique"]
    imageProductType = "ncc"
    embeddingType = "pencorr_20"
    plottingData = Metrics.load_BF_plotting_data(imageType=imageType, filters=filters,
                                                 imageProductType=imageProductType,
                                                 embeddingType=embeddingType)
    investigate_k(plottingData)
    investigate_BF_method(plottingData)
    plt.show()

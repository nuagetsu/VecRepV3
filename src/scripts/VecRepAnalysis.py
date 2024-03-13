import matplotlib.pyplot as plt

from data_processing import VecRep, FilepathUtils
from visualization import Metrics, GraphEstimates

imageType = "3bin"
filters = ["unique"]
imageProductType = "ncc"
embeddingType = "pencorr_20"
plottingDataFilepath = FilepathUtils.get_plotting_data_filepath(imageType=imageType, filters=filters,
                                                                imageProductType=imageProductType,
                                                                embeddingType=embeddingType)

plottingData = Metrics.load_plotting_data(plottingDataFilepath)

# Comparing the largest and the most negative eigenvalues
eigenFig, (ax1, ax2) = plt.subplots(2)
GraphEstimates.plot_eigenvalues(ax1, ax2, plottingData.initialEigenvalues, plottingData.finalEigenvalues)

# Seeing the neighbour scores of 3 random images TODO add average k among samples
scoreFig, ([imgAx1, ax1], [imgAx2, ax2], [imgAx3, ax3]) = plt.subplots(3, 2, width_ratios=[1,3])
axArr = [ax1, ax2, ax3]
imgAxArr = [imgAx1, imgAx2, imgAx3]
GraphEstimates.plot_k_neighbours(axArr, imgAxArr, plottingData.kNeighbourScores, plottingData.imagesFilepath)

# Display key values about the plot
displayText = ("Total Frobenius distance between imageProductMatrix and A^tA: " + "{:.2f}".format(plottingData.frobDistance) + "\n" +
               "Average Frobenius distance between imageProductMatrix and A^tA: " + "{:.2f}".format(plottingData.aveFrobDistance) + "\n" +
               "Max Frobenius distance between imageProductMatrix and A^tA: " + "{:.2f}".format(plottingData.maxDiff) + "\n")
statsPlot, ax = plt.subplots()
ax.text(0.5, 0.5, displayText, color='black',
        bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=1'), ha='center', va='center')
plt.show()


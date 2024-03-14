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

# Seeing the neighbour scores of 3 random images
scoreFig, ([imgAx1, ax1], [imgAx2, ax2], [imgAx3, ax3], [temp, aveAx]) = plt.subplots(4, 2, width_ratios=[1,3])
axArr = [ax1, ax2, ax3]
imgAxArr = [imgAx1, imgAx2, imgAx3]
temp.set_axis_off()
GraphEstimates.plot_k_neighbours(axArr=axArr, imageAxArr=imgAxArr, aveAx=aveAx, kNeighbourScores=plottingData.kNeighbourScores, imagesFilepath=plottingData.imagesFilepath)

# Display key values about the plot
statsPlot, ax = plt.subplots()
GraphEstimates.plot_key_stats_text(ax, plottingData)
plt.show()

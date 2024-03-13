import matplotlib.pyplot as plt
import src.visualization.GraphEstimates
import src.visualization.Metrics


imageType = "3bin"
filters = ["unique"]
imageProductType = "ncc"
embeddingType = "pencorr_20"
overwrite = {"filter": False, "im_prod": False, "estimate": False}
plottingData = Metrics.get_plotting_data(imageType=imageType, filters=filters, imageProductType=imageProductType,
                                         embeddingType=embeddingType)

# Comparing the largest and the most negative eigenvalues
fig, (ax1, ax2) = plt.subplots(2)
src.visualization.GraphEstimates.plot_eigenvalues(ax1, ax2, plottingData.initialEigenvalues, plottingData.finalEigenvalues)
plt.show()

plt.clf()
# Seeing the neighbour scores of 3 random images
scoreFig, (ax1, ax2, ax3) = plt.subplots(3)
imgFig, (imgAx1, imgAx2, imgAx3) = plt.subplots(3)
axArr = [ax1, ax2, ax3]
imgAxArr = [imgAx1, imgAx2, imgAx3]
src.visualization.GraphEstimates.plot_k_neighbours(axArr, imgAxArr, plottingData.kNeighbourScores, plottingData.imagesFilepath)
plt.show()

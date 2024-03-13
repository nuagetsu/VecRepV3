import matplotlib.pyplot as plt
import GraphEstimates
import Metrics


imageType = "3bin"
filters = ["unique"]
imageProductType = "ncc"
embeddingType = "pencorr_20"
overwrite = {"filter": False, "im_prod": False, "estimate": False}
plottingData = Metrics.get_plotting_data(imageType=imageType, filters=filters, imageProductType=imageProductType,
                                         embeddingType=embeddingType)

# Comparing the largest and the most negative eigenvalues
fig, (ax1, ax2) = plt.subplots(2)
GraphEstimates.plot_eigenvalues(ax1, ax2, plottingData.initialEigenvalues, plottingData.finalEigenvalues)
plt.show()

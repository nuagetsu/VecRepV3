import matplotlib.pyplot as plt

import src.data_processing.BruteForceEstimator as bfEstimator
import src.visualization.BFmethod as graphing

# -----Possible options-----

IMAGE_TYPES = ["NbinMmax_ones", "Nbin"]

"""
Nbin: N by N matrix of 1s and 0s

NbinMmax_ones: N by N matrix of 1s and 0s, with only M percentage of squares being 1s
"""

IMAGE_FILTERS = ["unique", "Nmax_ones", "one_island"]

"""
one_island: Outputs a set of images such that each image has only one connected island of 1s 
(Diagonals are not connected)

Pmax_ones: Outputs a set of images such that each image has only P or lower percentage of 1s 
Sample input: 60max_ones

unique: Outputs a set of images such that each image is NOT a simple translation of another image 
in the set
"""
IMAGE_PRODUCT_TYPES = ["ncc", "ncc_scaled"]
"""
ncc: Normal ncc score. Range of [0,1]
ncc_scaled: Normal ncc score, with range scaled to [-1,1]
"""

EMBEDDING_TYPES = ["pencorr_D"]

"""
pencorr_D: Find the nearest correlation matrix using pencorr, subject to the rank constraint.
Then computes embeddings with D dimensions, then normalize the embeddings before output
Sample input: pencorr_20
"""

# -----Variables-----
imageType = "3bin"
filters = ["unique"]
imageProductType = "ncc"
embeddingType = "pencorr_15"
overwrite = {"imgSet": False, "imgProd": False, "embedding": False}

# -----Execution-----

# Example to investigate k histograms
"""
bruteForceEstimator = bfEstimator.BruteForceEstimator(imageType=imageType, filters=filters, imageProductType=imageProductType,
                                                      embeddingType=embeddingType, overwrite=overwrite)
graphing.investigate_k(bruteForceEstimator)
"""

# Example to investigate a specific set of parameters for BF estimator
"""
bruteForceEstimator = bfEstimator.BruteForceEstimator(imageType=imageType, filters=filters, imageProductType=imageProductType,
                                                              embeddingType=embeddingType, overwrite=overwrite)
graphing.investigate_estimator(bruteForceEstimator, 16)
"""

# Example to investigate rank constraint
"""
graphing.investigate_BF_rank_constraint(imageType=imageType, filters=filters, imageProductType=imageProductType,
                                    startingConstr=5, endingConstr=10, specifiedKArr=[1, 3, 5], plotFrob=False)
"""
plt.show()

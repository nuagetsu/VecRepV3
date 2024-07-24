import matplotlib.pyplot as plt

import src.data_processing.BruteForceEstimator as bfEstimator
import src.visualization.BFmethod as graphing

# -----Possible options-----

IMAGE_TYPES = ["NbinMmax_ones", "Nbin", "triangle", "triangle_mean_subtracted"]

"""
Nbin: N by N matrix of 1s and 0s

NbinMmax_ones: N by N matrix of 1s and 0s, with only M percentage of squares being 1s

triangle: 8x8 matrix with a triangle contained in a 4x4 matrix within. Values are restricted to 0s and 1s.

triangle_mean_subtracted: The triangle type above with the mean of all entries subtracted from each entry. Values are
                        not restricted.
                        
triangle_gms: Triangle image set with mean subtracted across the whole image.
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
ncc_squared: Normal ncc score squared
"""

EMBEDDING_TYPES = ["pencorr_D"]

"""
pencorr_D: Find the nearest correlation matrix using pencorr, subject to the rank constraint.
Then computes embeddings with D dimensions, then normalize the embeddings before output
Sample input: pencorr_20
"""

# -----Variables-----
imageType = "triangles"
filters = ["unique"]
imageProductType = "ncc"
embeddingType = "pencorr_192"
overwrite = {"imgSet": False, "imgProd": False, "embedding": False}
weight = None

# -----Execution-----

# Example to investigate k histograms
"""
bruteForceEstimator = bfEstimator.BruteForceEstimator(imageType=imageType, filters=filters, imageProductType=imageProductType,
                                                      embeddingType=embeddingType, overwrite=overwrite, weight=weight)
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
                                    startingConstr=1, endingConstr=60, specifiedKArr=[3, 5],
                                    plotFrob=False, weight=None)
"""

# Example to investigate changes in image product
"""
graphing.investigate_image_product_type(imageType=imageType, filters=filters,
                                        imageProductTypeArr=["ncc", "ncc_pow_2", "ncc_pow_3"],
                                        embType=embeddingType, plotFrob=False, overwrite=overwrite)
"""

# Investigate changes in rank constraint for up to 5 different image products

graphing.investigate_BF_rank_constraint_for_image_types(imageType=imageType, filters=filters,
                                                        imageProductTypes=["ncc", "ncc_pow_2"],
                                                        startingConstr=1, endingConstr=192, interval=1,
                                                        specifiedKArr=[5],
                                                        progressive=False,
                                                        weights=["", ""],
                                                        embeddings=["pencorr", "pencorr"])

# Investigate changes in weight matrix for up to 9 different image products
"""
graphing.investigate_BF_weight_power(imageType=imageType, filters=filters,
                                     imageProductTypes=["ncc", "ncc_base_10"],
                                     startingConstr=0, endingConstr=20, interval=1,
                                     specifiedKArr=[5], plotFrob=False, rank=192)
"""

# Investigate changes in plateau rank as image set size increases
"""
graphing.investigate_plateau_rank_for_set_sizes(image_types=["shapes_3_4_dims_4_2", "shapes_3_4_5_dims_4_2", "shapes_3_5_dims_4_2", "shapes_3_dims_4_2", "shapes_4_dims_4_2", "shapes_5_dims_4_2", "shapes_6_dims_4_2", "shapes_3_4_6_dims_4_2", "shapes_3_6_dims_4_2", "shapes_4_6_dims_4_2", "shapes_5_6_dims_4_2", "shapes_4_5_6_dims_4_2", "shapes_3_4_5_6_dims_4_2", "shapes_3_5_6_dims_4_2", "shapes_7_dims_4_2", "shapes_3_7_dims_4_2", "shapes_4_7_dims_4_2", "shapes_5_7_dims_4_2", "shapes_6_7_dims_4_2", "shapes_3_4_7_dims_4_2", "shapes_3_5_7_dims_4_2", "shapes_3_6_7_dims_4_2", "shapes_4_5_7_dims_4_2", "shapes_4_6_7_dims_4_2", "shapes_5_6_7_dims_4_2", "shapes_3_5_6_7_dims_4_2", "shapes_3_4_5_7_dims_4_2", "shapes_3_4_6_7_dims_4_2", "shapes_4_5_6_7_dims_4_2", "shapes_3_4_5_6_7_dims_4_2"],
                                                filters=["unique"],
                                                image_product_types=["ncc", "ncc", "ncc", "ncc", "ncc", "ncc", "ncc", "ncc", "ncc", "ncc", "ncc", "ncc", "ncc", "ncc", "ncc", "ncc", "ncc", "ncc", "ncc", "ncc", "ncc", "ncc", "ncc", "ncc", "ncc", "ncc", "ncc", "ncc", "ncc", "ncc"],
                                                embeddings=["pencorr_python", "pencorr_python", "pencorr_python", "pencorr_python", "pencorr_python", "pencorr_python", "pencorr_python", "pencorr_python", "pencorr_python", "pencorr_python", "pencorr_python", "pencorr_python", "pencorr_python", "pencorr_python", "pencorr_python", "pencorr_python", "pencorr_python", "pencorr_python", "pencorr_python", "pencorr_python", "pencorr_python", "pencorr_python", "pencorr_python", "pencorr_python", "pencorr_python", "pencorr_python", "pencorr_python", "pencorr_python", "pencorr_python", "pencorr_python"],
                                                weights=["", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""],
                                                k=5, prox=3, overwrite=None)
"""
"""
graphing.investigate_plateau_rank_for_set_sizes(image_types=["randomshapes_3_4_5_dims_10_3_500", "randomshapes_3_4_5_dims_10_3_600", "randomshapes_3_4_5_dims_10_3_700", "randomshapes_3_4_5_dims_10_3_800", "randomshapes_3_4_5_dims_10_3_900", "randomshapes_3_4_5_dims_10_3_1000", "randomshapes_3_4_5_dims_10_3_1200", "randomshapes_3_4_5_dims_10_3_1400", "randomshapes_3_4_5_dims_10_3_1600", "randomshapes_3_4_5_dims_10_3_1800", "randomshapes_3_4_5_dims_10_3_2000", "randomshapes_3_4_5_dims_10_3_2500", "randomshapes_3_4_5_dims_10_3_3000", "randomshapes_3_4_5_dims_10_3_3500", "randomshapes_3_4_5_dims_10_3_4000"],
                                                filters=["unique"],
                                                image_product_types=["ncc", "ncc", "ncc", "ncc", "ncc", "ncc", "ncc", "ncc", "ncc", "ncc", "ncc", "ncc", "ncc", "ncc", "ncc"],
                                                embeddings=["pencorr_python", "pencorr_python", "pencorr_python", "pencorr_python", "pencorr_python", "pencorr_python", "pencorr_python", "pencorr_python", "pencorr_python", "pencorr_python", "pencorr_python", "pencorr_python", "pencorr_python", "pencorr_python", "pencorr_python"],
                                                weights=["", "", "", "", "", "", "", "", "", "", "", "", "", "", ""],
                                                k=5, prox=3, overwrite=None)
"""


# Investigate changes in plateau rank as the image set size remains constant but image size increases
"""
graphing.investigate_plateau_rank_for_image_sizes(image_types=["randomshapes_3_4_dims_4_2_800", "randomshapes_3_4_dims_5_2_800", "randomshapes_3_4_dims_6_2_800", "randomshapes_3_4_dims_7_2_800", "randomshapes_3_4_dims_8_2_800", "randomshapes_3_4_dims_9_2_800", "randomshapes_3_4_dims_10_2_800", "randomshapes_3_4_dims_11_2_800", "randomshapes_3_4_dims_12_2_800", "randomshapes_3_4_dims_4_2_1500", "randomshapes_3_4_dims_5_2_1500", "randomshapes_3_4_dims_6_2_1500", "randomshapes_3_4_dims_7_2_1500", "randomshapes_3_4_dims_8_2_1500", "randomshapes_3_4_dims_9_2_1500", "randomshapes_3_4_dims_10_2_1500", "randomshapes_3_4_dims_11_2_1500", "randomshapes_3_4_dims_12_2_1500"],
                                                  filters=["unique"],
                                                  image_product_types=["ncc", "ncc", "ncc", "ncc", "ncc", "ncc", "ncc", "ncc", "ncc", "ncc", "ncc", "ncc", "ncc", "ncc", "ncc", "ncc", "ncc", "ncc"],
                                                  embeddings=["pencorr_python", "pencorr_python", "pencorr_python", "pencorr_python", "pencorr_python", "pencorr_python", "pencorr_python", "pencorr_python", "pencorr_python", "pencorr_python", "pencorr_python", "pencorr_python", "pencorr_python", "pencorr_python", "pencorr_python", "pencorr_python", "pencorr_python", "pencorr_python"],
                                                  weights=["", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""],
                                                  k=5, prox=3, overwrite=None)
"""

plt.show()

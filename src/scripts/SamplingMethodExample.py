import matplotlib.pyplot as plt
import numpy as np

from src.data_processing.ImageGenerators import get_island_image_set
from src.data_processing.ImageGenerators import get_triangle_image_set
from src.visualization import SamplingMethod, BFmethod
from src.data_processing.SampleTester import SampleTester
from src.data_processing.SampleEstimator import SampleEstimator

"""
Some example graphs for the sampling method randomly generated image set
"""

# -----Variables-----
imageType = "triangle"
"""
N island M max_ones: Generates a random island in a N by N matrix with up to M max ones, e.g. 10island30max_ones

triangle: 8x8 matrix with a triangle contained in a 4x4 matrix within.

triangle_mean_subtracted: triangle image set with mean subtracted from each triangle.
"""

imageProductType = "ncc_base_10_rep_2"
testSize = 42
trainingSize = 150
embeddingType = "pencorr_30"
specifiedKArr = [5]
sampleName = "Triangle mean example sample"
testName = "Triangle mean example test"

# Loading image dataset. Training set takes from random samples of the image set.

if imageType == "triangle":
    imageSet = get_triangle_image_set()
elif imageType == "triangle_mean_subtracted":
    imageSet = get_triangle_image_set(mean_subtracted=True)
else:
    imageSet = get_island_image_set(imageType, 500)
imageSet = np.array(imageSet)
testSample, trainingSample = SamplingMethod.generate_random_sample(imageSet, testSize, trainingSize)

# -----Creating graphs-----
"""
# Example of investigating a specific set of parameters

sampleEstimator = SampleEstimator(sampleName=sampleName, trainingImageSet=trainingSample, embeddingType=embeddingType,
                                  imageProductType=imageProductType)
sampleTester = SampleTester(testImages=testSample, sampleEstimator=sampleEstimator, testName=testName)
BFmethod.investigate_k(sampleTester)
BFmethod.investigate_estimator(sampleTester)
"""
# Example of sweeping the size of the training data set
"""
startingTrainingSize = 100
endingTrainingSize = 190
increment = 10

SamplingMethod.investigate_training_size(imageSet=imageSet, imageProductType=imageProductType,
                                         embeddingType=embeddingType, startingTrainingSize=startingTrainingSize,
                                         endingTrainingSize=endingTrainingSize, increment=increment, testSize=testSize,
                                         testPrefix=testName, specifiedKArr=specifiedKArr, trials=5)
"""

# Example of sweeping the rank constraint of the estimator
"""
max_size = len(imageSet)
sampleSize = 200
if sampleSize > max_size:
    sampleSize = max_size

SamplingMethod.investigate_tester_rank_constraint(imageSet=imageSet, imageProductType=imageProductType,
                                                  sampleSize=sampleSize, testSize=testSize, testPrefix=testName,
                                                  startingConstr=5, endingConstr=50, increment=5)
"""
# Example of sweeping the size of the training data set for multiple image product types

startingTrainingSize = 100
endingTrainingSize = 190
increment = 10

SamplingMethod.investigate_training_size_for_image_products(imageSet=imageSet,
                                                            imageProductTypes=["ncc", "ncc_pow_2", "ncc_base_10"],
                                                            embeddingType=embeddingType,
                                                            startingTrainingSize=startingTrainingSize,
                                                            endingTrainingSize=endingTrainingSize, increment=increment,
                                                            testSize=testSize,
                                                            testPrefix=testName, specifiedKArr=specifiedKArr, trials=5,
                                                            weights=["", "", ""])


plt.show()

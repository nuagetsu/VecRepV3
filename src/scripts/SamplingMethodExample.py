import matplotlib.pyplot as plt
import numpy as np

from src.data_processing.ImageGenerators import get_island_image_set
from src.visualization import SamplingMethod, BFmethod
from src.data_processing.SampleTester import SampleTester
from src.data_processing.SampleEstimator import SampleEstimator

"""
Some example graphs for the sampling method randomly generated image set
"""

# -----Variables-----
imageType = "10island30max_ones"
"""
N island M max_ones: Generates a random island in a N by N matrix with up to M max ones
"""

imageProductType = "ncc"
testSize = 100
trainingSize = 300
embeddingType = "pencorr_30"
specifiedKArr = [5]
sampleName = "Bigger example sample"
testName = "Bigger example test"

# Loading image dataset. Training set takes from random samples of the image set.
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

startingTrainingSize = 50
endingTrainingSize = 500
increment = 50
"""
SamplingMethod.investigate_training_size(imageSet=imageSet, imageProductType=imageProductType,
                                         embeddingType=embeddingType, startingTrainingSize=startingTrainingSize,
                                         endingTrainingSize=endingTrainingSize, increment=increment, testSize=testSize,
                                         testPrefix=testName, specifiedKArr=specifiedKArr)

"""
# Example of sweeping the rank constraint of the estimator

SamplingMethod.investigate_tester_rank_constraint(imageSet=imageSet, imageProductType=imageProductType, sampleSize=200,
                                                  testSize=testSize, testPrefix=testName, startingConstr=5,
                                                  endingConstr=30, increment=5)

plt.show()

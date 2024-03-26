from src.data_processing.ImageGenerators import get_island_image_set
import numpy as np
import matplotlib.pyplot as plt
from src.data_processing.SampleTester import SampleTester
from src.data_processing.SampleEstimator import SampleEstimator
from src.visualization import BFmethod, SamplingMethod
"""
Some example graphs for the sampling method using a pre-generated image set
"""

# -----Creating the dataset-----


arr = get_island_image_set("5island30max_ones", 500)
arr = np.array(arr)
np.save("toy_sample", arr)


# -----Variables-----
imageType = "5island30max_ones"
"""
N island M max_ones: Generates a random island in a N by N matrix with up to M max ones
"""

imageProductType = "ncc"
testSize = 50
trainingSize = 100
embeddingType = "pencorr_30"
specifiedKArr = [1, 3, 5]
sampleName = "Example sample"
testName = "Example test"

# Loading image dataset. Training set takes from the front of the image set and test set takes from the end
imageSet = np.load("toy_sample.npy")
trainingSample = imageSet[trainingSize:]
testSample = imageSet[-testSize:]

# -----Creating graphs-----

# Example of investigating a specific set of parameters
"""
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

SamplingMethod.investigate_training_size(imageSet=imageSet, imageProductType=imageProductType,
                                         embeddingType=embeddingType, startingTrainingSize=startingTrainingSize,
                                         endingTrainingSize=endingTrainingSize, increment=increment, testSize=testSize,
                                         testPrefix=testName, specifiedKArr=specifiedKArr)



plt.show()

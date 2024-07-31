import matplotlib.pyplot as plt
import numpy as np
from line_profiler import profile

from src.data_processing.ImageGenerators import get_island_image_set
from src.data_processing.ImageGenerators import get_triangles_image_set
from src.data_processing.ImageGenerators import get_quadrilaterals_image_set
from src.data_processing.Utilities import generate_filtered_image_set
from src.helpers.FilepathUtils import get_image_set_filepath
from src.visualization import SamplingMethod, BFmethod
from src.data_processing.SampleTester import SampleTester
from src.data_processing.SampleEstimator import SampleEstimator

"""
Some example graphs for the sampling method randomly generated image set
"""

# -----Variables-----
imageType = "quadrilaterals"
"""
N island M max_ones: Generates a random island in a N by N matrix with up to M max ones, e.g. 10island30max_ones

triangles: 8x8 matrix with a triangle contained in a 4x4 matrix within.

"""

imageProductType = "ncc_base_10_rep_2"
weight = ""
filters = []
testSize = 42
trainingSize = 150
embeddingType = "pencorr_50"
specifiedKArr = [5]
sampleName = "Triangle example sample"
testName = "Triangle example test"

# Loading image dataset. Training set takes from random samples of the image set.

imageSet = generate_filtered_image_set(imageType, filters, get_image_set_filepath(imageType, filters))
imageSet = np.array(imageSet)
# testSample, trainingSample = SamplingMethod.generate_random_sample(imageSet, testSize, trainingSize)

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

SamplingMethod.investigate_training_size(imageSet=imageSet, imageProductType=imageProductType, weight=weight
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

SamplingMethod.investigate_tester_rank_constraint(imageSet=imageSet, imageProductType=imageProductType, weight=weight
                                                  sampleSize=sampleSize, testSize=testSize, testPrefix=testName,
                                                  startingConstr=5, endingConstr=50, increment=5)
"""
# Example of sweeping the size of the training data set for multiple image product types
"""
startingTrainingSize = 100
endingTrainingSize = 190
increment = 10

SamplingMethod.investigate_training_size_for_image_products(imageSet=imageSet,
                                                            imageProductTypes=["ncc", "ncc_base_10", "ncc_base_10_rep_2"],
                                                            embeddingType=embeddingType,
                                                            startingTrainingSize=startingTrainingSize,
                                                            endingTrainingSize=endingTrainingSize, increment=increment,
                                                            testSize=testSize,
                                                            testPrefix=testName, specifiedKArr=specifiedKArr, trials=5,
                                                            weights=["", "", ""])
"""
# Example of sweeping the rank constraint of the estimator with multiple image products
"""
max_size = len(imageSet)
sampleSize = 300
if sampleSize > max_size:
    sampleSize = max_size
testSize = 50

imageType = "quadrilaterals"
sampleName = "Quadrilaterals example sample"
testName = "Quadrilaterals example test"

SamplingMethod.investigate_tester_rank_constraint_for_image_products(imageSet=imageSet,
                                                                     imageProductTypes=["ncc", "ncc"],
                                                                     sampleSize=sampleSize, testSize=testSize,
                                                                     testPrefix=testName, startingConstr=5,
                                                                     endingConstr=300, increment=1,
                                                                     weights=["", ""],
                                                                     embeddings=["pencorr", "dblcorr"],
                                                                     trials=1, progressive=True)
"""
# Example of sweeping the rank constraint of the estimator with multiple image products while specifying training and
# test image sets
"""
sampleName = "Shapes example sample"
testName = "Shapes example test"

SamplingMethod.investigate_sample_and_test_sets(trainingSet="shapes_3_4_dims_4_2", testSet="ramdomshapes_3_4_dims_4_2_80", filters=["unique"],
                                                trainingSize=600, testSize=80, imageProductTypes=["ncc_pow_2"],
                                                weights=["ncc_factor_1"], startingConstr=10, endingConstr=11,
                                                increment=1, progressive=True, trials=1, embeddings=["pencorr"],
                                                testPrefix=testName)
"""

# Example of finding plateau rank for a sample
sampleName = "Shapes example sample"
test_prefix = "Shapes example test"
training_sets = ["shapes_3_dims_4_2"]
test_sets = ["shapes_3_dims_4_2"]
training_sizes = [150]
image_product_types = ["ncc"]
weights = [""]
embeddings = ["pencorr_python"]
filters = ["unique"]

@profile
def main():
    SamplingMethod.investigate_sample_plateau_rank(training_sets=training_sets, test_set=test_sets,
                                                   training_sizes= training_sizes, image_product_types=image_product_types,
                                                   weights=weights, embeddings=embeddings, filters=filters, test_prefix=test_prefix,
                                                   prox=3, trials=1, k=5)
    # print(df)

if __name__ == "__main__":
    main()

plt.show()

import numpy as np

import time
from data_processing import FilepathUtils
from data_processing.VecRep import generate_filtered_image_set, generate_image_product_matrix, \
    generate_embedding_matrix, generate_plotting_data


def get_sample_name(sampleSize):
    timestr = time.strftime("%Y%m%d-%H%M%S")
    sampleName = str(sampleSize) + "_size_sample_on_" + timestr
    return sampleName


class SampleEstimator:
    def __init__(self, *, sampleSize: int, imageType: str, filters, embeddingType: str, imageProductType: str):
        self.sampleSize = sampleSize
        self.sampleName = get_sample_name(sampleSize)
        self.imageType = imageType
        self.filters = filters
        self.embeddingType = embeddingType
        self.imageProductType = imageProductType
        self.sampleDirectory = FilepathUtils.get_sample_directory(imageType=imageType, filters=filters,
                                                                  imageProductType=imageProductType,
                                                                  embeddingType=embeddingType, sampleName=self.sampleName)

        print("Generating filtered images....")
        imageSetFilepath = FilepathUtils.get_image_set_filepath(imageType=imageType, filters=filters)
        imageSet = generate_filtered_image_set(imageType=imageType, filters=filters,
                                               imageSetFilepath=imageSetFilepath)

        # Sampling the image set
        self.sampledImageSet = random_sample_from_array(imageSet, sampleSize)
        sampledImageSetFilepath = FilepathUtils.get_sample_images_filepath(self.sampleDirectory)
        np.save(sampledImageSetFilepath, self.sampledImageSet)

        print("Generating image product matrix....")
        imageProductFilepath = FilepathUtils.get_sample_ipm_filepath(self.sampleDirectory)
        self.imageProductMatrix = generate_image_product_matrix(imageSet=self.sampledImageSet,
                                                                imageProductType=imageProductType,
                                                                imageProductFilepath=imageProductFilepath)

        print("Generating embeddings....")
        embeddingFilepath = FilepathUtils.get_sample_embedding_filepath(self.sampleDirectory)
        self.embeddingMatrix = generate_embedding_matrix(imageProductMatrix=self.imageProductMatrix,
                                                         embeddingType=embeddingType,
                                                         embeddingFilepath=embeddingFilepath)
        print("Saving plotting data....")
        plottingDataFilepath = FilepathUtils.get_sample_plotting_data_filepath(self.sampleDirectory)
        generate_plotting_data(plottingDataFilepath=plottingDataFilepath, imageProductMatrix=self.imageProductMatrix,
                               embeddingMatrix=self.embeddingMatrix, imagesFilepath=imageSetFilepath)

    def get_embedding_estimate(self, imageInput):
        pass



def random_sample_from_array(arr, n):
    """
    Generates a random sample of n elements from a 1D NumPy array.

    Args:
        arr (numpy.ndarray): Input 1D array.
        n (int): Number of elements to sample.

    Returns:
        numpy.ndarray: Random sample of n elements from the input array.
    """
    if n >= len(arr):
        raise ValueError("n must be less than the length of the array.")

    # Generate random indices for sampling
    random_indices = np.random.choice(len(arr), size=n, replace=False)

    # Create the sampled array
    sampled_array = arr[random_indices]

    return sampled_array

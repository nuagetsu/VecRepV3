import os
from pathlib import Path
from typing import List


def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent


def get_filepath(*, imageType: str, filters=None, imageProductType=None, embeddingType=None):
    """
    :return: The path of the directory where this information should be saved.
    """
    dirName = imageType
    if filters is not None and filters:
        filterName = ""
        for filter in filters:
            filterName = filterName + filter + "--"
        filterName = filterName[:-2]
        dirName = dirName + "-" + filterName  # Directory name is <imageType>-<filter1>--<filter2>--<filter3>
    filepath = os.path.join(get_project_root(), "data", dirName)
    if imageProductType is not None:
        filepath = os.path.join(filepath, imageProductType)
        if embeddingType is not None:
            filepath = os.path.join(filepath, embeddingType)
    return filepath


def get_matlab_dirpath() -> str:
    return os.path.join(get_project_root(), "src", "matlab_functions")


def get_image_set_filepath(imageType: str, filters: List[str]) -> str:
    filepath = get_filepath(imageType=imageType, filters=filters)
    return os.path.join(filepath, "filtered_images.npy")


def get_image_product_filepath(imageType: str, filters: List[str], imageProductType: str) -> str:
    filepath = get_filepath(imageType=imageType, filters=filters, imageProductType=imageProductType)
    return os.path.join(filepath, "image_product_matrix")


def get_embedding_matrix_filepath(imageType: str, filters: List[str], imageProductType: str, embeddingType: str,
                                  weight=None) -> str:
    filepath = get_filepath(imageType=imageType, filters=filters, imageProductType=imageProductType,
                            embeddingType=embeddingType)
    if weight is None or weight == "":
        weight = "unweighted"
    return os.path.join(filepath, weight, "embedding_matrix")


def get_weighting_matrix_filepath(imageType: str, filters: List[str], weightingType: str, copy="") -> str:
    """
    :param imageType: Image set name
    :param filters: Filters used
    :param weightingType: Type of weighting matrix
    :param copy: Whether a copy of G is used as the weighting matrix
    :return: Filepath of weighting matrix
    """
    filepath = get_filepath(imageType=imageType, filters=filters)

    components = weightingType.split("_factor_")
    if components[0] == "copy":
        weightingType = copy + "_factor_" + components[1]

    return os.path.join(filepath, "weights", weightingType)


def get_plotting_data_filepath(imageType: str, filters: List[str], imageProductType: str, embeddingType: str) -> str:
    filepath = get_filepath(imageType=imageType, filters=filters, imageProductType=imageProductType,
                            embeddingType=embeddingType)
    return os.path.join(filepath, "plotting_data")


def get_sample_directory(sampleName: str, category="uncategorized") -> str:
    """
    :param sampleName: Name of sample
    :return: Directory where the sample data should be saved
    """
    return os.path.join(get_project_root(), "data", "samples", category, sampleName)


def get_set_size_df_filepath(image_types: list):
    """
    Function to get a filepath to save a dataframe in the dataframe folder of data. Dataframes are not used for
    any particular function yet but can be viewed for analysis
    :param image_types: Image types analyzed
    :return: Dataframe containing data for analyzed image types.

    """
    number = len(image_types)
    name = str(number) + " types of size "
    split = image_types[0].split("_")
    dim_index = split.index("dims")
    shape_size = split[dim_index + 1]
    image_size = split[dim_index + 2]
    name += str(shape_size) + " in " + str(image_size) + ".csv"
    return os.path.join(get_project_root(), "data", "dataframes", name)



def get_sample_embedding_matrix_filepath(embeddingType, sampleDirectory: str, weight=None):
    """
    :param embeddingType: Type of embedding used
    :param sampleDirectory: Directory of sample
    :param weight: Weighting matrix type
    :return: Filepath for embedding matrix for sampling method
    """
    if weight is None or weight == "":
        weight = "unweighted"
    return os.path.join(sampleDirectory, embeddingType, weight, "sample_embeddings")


def get_sample_weighting_filepath(sampleDirectory: str, weightingType: str, copy="") -> str:
    """
    :param sampleDirectory: Directory/filepath of sample
    :param weightingType: Weighting matrix type
    :param copy: Whether G is used as weighting matrix
    :return: Filepath for weighting matrix when used in sampling method
    """
    components = weightingType.split("_factor_")
    if components[0] == "copy":
        weightingType = copy + "_factor_" + components[1]

    return os.path.join(sampleDirectory, "weightings", weightingType)


def get_sample_ipm_filepath(sampleDirectory: str):
    return os.path.join(sampleDirectory, "sample_image_product_matrix")


def get_sample_images_filepath(sampleDirectory: str):
    return os.path.join(sampleDirectory, "sample_images.npy")


def get_sample_plotting_data_filepath(sampleDirectory: str):
    return os.path.join(sampleDirectory, "sample_plotting_data")


def get_sample_info_filepath(sampleDirectory: str):
    return os.path.join(sampleDirectory, "sample_info")


def get_test_images_filepath(sampleDirectory: str, testName: str):
    return os.path.join(sampleDirectory, testName, "test_image_set.npy")


def get_test_embeddings_filepath(sampleDirectory: str, testName: str):
    return os.path.join(sampleDirectory, testName, "test_embeddings")


def get_test_ipm_filepath(sampleDirectory: str, testName: str):
    return os.path.join(sampleDirectory, testName, "test_image_product_matrix")


def get_matching_sample_filepath(matching_sample_name: str):
    return os.path.join(get_project_root(), "data", "matching", matching_sample_name, "matching_images.npy")


def get_full_matching_image_set_filepath(matching_directory: str, training_sample_name: str):
    matching_directory = os.path.split(matching_directory)[0]
    return os.path.join(matching_directory, training_sample_name, "full_images.npy")


def get_matching_embeddings_filepath(matching_directory: str, image_product_type: str, embedding_type: str):
    return os.path.join(matching_directory, image_product_type, embedding_type, "embeddings.npy")

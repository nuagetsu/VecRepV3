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


def get_embedding_matrix_filepath(imageType: str, filters: List[str], imageProductType: str, embeddingType: str) -> str:
    filepath = get_filepath(imageType=imageType, filters=filters, imageProductType=imageProductType,
                            embeddingType=embeddingType)
    return os.path.join(filepath, "embedding_matrix")


def get_plotting_data_filepath(imageType: str, filters: List[str], imageProductType: str, embeddingType: str) -> str:
    filepath = get_filepath(imageType=imageType, filters=filters, imageProductType=imageProductType,
                            embeddingType=embeddingType)
    return os.path.join(filepath, "plotting_data")


def get_sample_directory(sampleName:str) -> str:
    """
    :param sampleName: Name of sample
    :return: Directory where the sample data should be saved
    """
    return os.path.join(get_project_root(), "data", "samples", sampleName)


def get_sample_embedding_matrix_filepath(sampleDirectory: str):
    return os.path.join(sampleDirectory, "sample_embeddings")


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



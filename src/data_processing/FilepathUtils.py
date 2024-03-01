import os
from pathlib import Path


def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent


def get_filepath(*, imageType: str, filters=None, imageProductType=None, embeddingType=None):
    """
    :return: The path of the directory where this information should be saved. Also generates the directory if it
    does not already exist
    """
    dirName = imageType
    if filters is not None:
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

def get_image_set_filepath(*, imageType: str, filters=None) -> str:
    filepath = get_filepath(imageType=imageType, filters=filters)
    return os.path.join(filepath, "filtered_images.npy")


def get_image_product_filepath(*, imageType: str, filters=None, imageProductType: str) -> str:
    filepath = get_filepath(imageType=imageType, filters=filters, imageProductType=imageProductType)
    return os.path.join(filepath, "image_product_matrix")


def get_embedding_matrix_filepath(*, imageType: str, filters=None, imageProductType: str, embeddingType: str) -> str:
    filepath = get_filepath(imageType=imageType, filters=filters, imageProductType=imageProductType,
                            embeddingType=embeddingType)
    return os.path.join(filepath, "embedding_matrix")


def get_plotting_data_filepath(*, imageType: str, filters=None, imageProductType: str, embeddingType: str) -> str:
    filepath = get_filepath(imageType=imageType, filters=filters, imageProductType=imageProductType,
                            embeddingType=embeddingType)
    return os.path.join(filepath, "plotting_data")
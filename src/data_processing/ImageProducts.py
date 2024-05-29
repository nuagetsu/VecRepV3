import re
from typing import Callable

import cv2
import numpy as np
from numpy.typing import NDArray


def get_image_product(imageProductType: str):
    if imageProductType == "ncc":
        return ncc
    if imageProductType == "ncc_scaled":
        return ncc_scaled
    elif imageProductType == "ncc_squared":
        return ncc_squared
    elif imageProductType == "ncc_squared_scaled":
        return ncc_squared_scaled
    elif re.search("ncc_pow_[0-9]*[0-9]\.?\d*$", imageProductType) is not None:
        power = float(re.search(r"[0-9]*[0-9]\.?\d*", imageProductType).group())
        return ncc_pow(power)
    else:
        raise ValueError(imageProductType + " is not a valid image product type")

def ncc_scaled(mainImg: NDArray, tempImg: NDArray) -> float:
    """
    :param mainImg: Main image to be scanned
    :param tempImg: Template image to be scanned over the main
    :return: Max value of the ncc, with scaled bounds of [-1,1]
    """
    return ncc(mainImg, tempImg) * 2 - 1

def ncc_squared(mainImg: NDArray, tempImg: NDArray) -> float:
    """
    :param mainImg: Main image to be scanned
    :param tempImg: Template image to be scanned over the main
    :return: Max value of the ncc, squared (to keep 1 to 1)

    In theory, this should further separate close images with high NCC score that we care more about.
    """
    return ncc(mainImg, tempImg) ** 2

def ncc_squared_scaled(mainImg: NDArray, tempImg: NDArray) -> float:
    """
    :param mainImg: Main image to be scanned
    :param tempImg: Template image to be scanned over the main
    :return: Max value of the ncc squared, with scaled bounds of [-1,1]
    """
    return ncc_squared(mainImg, tempImg) * 2 - 1

def ncc_pow(power: float):
    """
    :param power: Power to raise NCC score by
    :return: Image product method of ncc raised to the power of power
    """
    return lambda mainImg, tempImg: ncc(mainImg, tempImg) ** power

def ncc(mainImg: NDArray, tempImg: NDArray) -> float:
    """
    :param mainImg: Main image to be scanned
    :param tempImg: Template image to be scanned over the main
    :return: Max value of the ncc

    Applies NCC of the template image over the main image and returns the max value obtained.
    When the template image kernel exceeds the bounds, wraps to the other side of the main image
    """
    if np.count_nonzero(mainImg) == 0:
        if np.count_nonzero(tempImg) == 0:
            return 1
        return 0

    mainImg = np.pad(mainImg, max(len(mainImg), len(mainImg[0])),
                     'wrap')  # Padding the main image with wrapped values to simulate wrapping

    mainImg = np.asarray(mainImg, np.single)  # Setting data types of array
    tempImg = np.asarray(tempImg, np.single)

    corr = cv2.matchTemplate(mainImg, tempImg, cv2.TM_CCORR_NORMED)

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(corr)

    return max_val

def calculate_image_product_matrix(imageSet: NDArray, imageProduct: Callable) -> NDArray:
    """
    Applies the image product between every possible permutation of images in the imageSet
    """
    imageProductMatrix = []
    for image1 in imageSet:
        for image2 in imageSet:
            imageProductMatrix.append(imageProduct(image1, image2))
    imageProductMatrix = np.reshape(imageProductMatrix, (len(imageSet), len(imageSet)))
    return imageProductMatrix

def calculate_image_product_vector(newImage: NDArray, imageSet: NDArray, imageProduct: Callable):
    """
    :param newImage: New image which you want to find the image product vector of
    :param imageSet: Images to be comapared to
    :param imageProduct: image product used to compare
    :return: A 1d numpy array which is the image product of the new image with each of the images in the imageset
    """
    if newImage.shape != imageSet[0].shape:
        raise ValueError("Input image has the dimensions " + str(newImage.shape) + " when it should be "
                         + str(imageSet[0].shape))
    imageProductVector = []
    for image in imageSet:
        imageProductVector.append(imageProduct(newImage, image))
    imageProductVector = np.array(imageProductVector)
    return imageProductVector

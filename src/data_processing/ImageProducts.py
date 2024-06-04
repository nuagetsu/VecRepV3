import re
from typing import Callable

import cv2
import numpy as np
from numpy.typing import NDArray
import math


def get_image_product(imageProductType: str):
    if imageProductType == "ncc":
        return ncc
    if imageProductType == "ncc_scaled":
        return ncc_scaled
    elif re.search(r"ncc_pow_[0-9]+\.?\d*$", imageProductType) is not None:
        power = float(re.search(r"[0-9]+?\.?\d*", imageProductType).group())
        return image_product_pow(ncc, power)
    elif imageProductType == "ncc_exp":
        return image_product_exp_repeated(ncc, 1)
    elif re.search(r"ncc_exp_pow_[0-9]+\.?\d*$", imageProductType) is not None:
        power = float(re.search(r"[0-9]+?\.?\d*", imageProductType).group())
        return image_product_pow(image_product_exp_repeated(ncc, 1), power)
    elif re.search("ncc_exp_rep_[0-9]+$", imageProductType) is not None:
        reps = int(re.search(r"[0-9]+", imageProductType).group())
        return image_product_exp_repeated(ncc, reps)
    elif imageProductType == "ncc_log":
        return image_product_log_base_2(ncc, 1)
    elif re.search(r"ncc_log_rep_[0-9]+$", imageProductType) is not None:
        reps = int(re.search(r"[0-9]+", imageProductType).group())
        return image_product_log_base_2(ncc, reps)
    elif re.search(r"ncc_base_[0-9]+\.?\d*$", imageProductType) is not None:
        base = float(re.search(r"[0-9]+\.?\d*", imageProductType).group())
        return image_product_as_power(ncc, base, 1)
    elif re.search(r"ncc_base_[0-9]+\.?\d*_rep_[0-9]+$", imageProductType) is not None:
        matches = re.findall(r"[0-9]+\.?\d*", imageProductType)
        base = float(matches[0])
        reps = int(matches[1])
        return image_product_as_power(ncc, base, reps)
    else:
        raise ValueError(imageProductType + " is not a valid image product type")

def ncc_scaled(mainImg: NDArray, tempImg: NDArray) -> float:
    """
    :param mainImg: Main image to be scanned
    :param tempImg: Template image to be scanned over the main
    :return: Max value of the ncc, with scaled bounds of [-1,1]
    """
    return ncc(mainImg, tempImg) * 2 - 1

def image_product_pow(image_product, power: float):
    """
    :param image_product: Image product to be modified
    :param power: Power to raise the image product by
    :return: Image product method of initial image product raised to the power of power

    In theory, this should further separate close images with high NCC score that we care more about.
    """
    return lambda mainImg, tempImg: image_product(mainImg, tempImg) ** power

def image_product_exp_repeated(image_product, reps: int):
    """
    :param image_product: Image product to be modified
    :param reps: Number of times to repeat the function
    :return: Value of e raised to the power of n - 1, repeated the number of times indicated
    """
    def res(mainImage, tempImg):
        func = image_product(mainImage, tempImg)
        for i in range(0, reps):
            func = math.exp(func - 1)
        return func
    return res

def image_product_log_base_2(image_product, reps: int):
    """
    :param image_product: Image product to be modified
    :param reps: Number of times the modification should be repeated
    :return: Value of log base 2 of 1 + result of image product, repeated the number of times indicated
    """
    def res(mainImage, tempImg):
        func = image_product(mainImage, tempImg)
        for i in range(0, reps):
            func = math.log2(1 + func)
        return func
    return res

def image_product_as_power(image_product, base: float, reps: int):
    """
    :param image_product: Image product to be modified
    :param base: Base to which will be raised by (image product - 1)
    :param reps: Number of times the modification should be repeated
    :return: Base raised by (image product - 1)
    """
    def res(mainImage, tempImg):
        func = image_product(mainImage, tempImg)
        for i in range(0, reps):
            func = base ** (func - 1)
        return func
    return res

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

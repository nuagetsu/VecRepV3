import re
from typing import Callable

import cv2
import numpy as np
from numpy.typing import NDArray
import math


def get_image_product(imageProductType: str):
    """
    Converts an image product type string into an image product function modified with monotonic transformations.
    :param imageProductType: image product type string
    :return: image product function
    """
    elems = imageProductType.split("_")
    func = get_base_image_product(elems[0])
    length = int((len(elems) - 1) / 2)
    if len(elems) == 2:
        raise ValueError(imageProductType + ' is of the incorrect format! Separate each modifier with a "_"!')
    for i in range(0, length):
        func = edit_image_product(func, elems[1 + 2 * i], float(elems[2 + 2 * i]))
    return func


def get_base_image_product(imageProductType: str):
    """
    :param imageProductType: Function to get the base image product to be further modified through monotonic
    transformations.
    :return: The base image product

    Currently implemented image products: "ncc"
    """
    if imageProductType == "ncc":
        return ncc
    else:
        raise ValueError(imageProductType + " is not a valid image product type!")


def edit_image_product(imageProduct, mod: str, factor=None):
    if mod == "scaled":
        return scale_min(imageProduct, factor)
    elif mod == "pow":
        return image_product_pow(imageProduct, factor)
    elif mod == "exp":
        return image_product_exp_repeated(imageProduct, 1)
    elif mod == "log":
        return image_product_log(imageProduct, factor)
    elif mod == "base":
        return image_product_as_power(imageProduct, factor)
    elif mod == "mult":
        return multiply_image_product(imageProduct, factor)
    else:
        raise ValueError(mod + " is not a valid modification!")


def ncc_scaled(mainImg: NDArray, tempImg: NDArray) -> float:
    """
    :param mainImg: Main image to be scanned
    :param tempImg: Template image to be scanned over the main
    :return: Max value of the ncc, with scaled bounds of [-1,1]
    """
    return ncc(mainImg, tempImg) * 2 - 1


def scale_min(image_product, min_value: float):
    """
    :param image_product: Image product to be modified
    :param min_value: Minimum value to scale
    :return: Image product scaled from the minimum value to 1
    """
    if not -1 <= min_value < 1:
        raise ValueError("Minimum value to scale must be in range [-1, 1)!")
    factor = 1 - min_value
    return lambda mainImg, tempImg: image_product(mainImg, tempImg) * factor + min_value


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


def image_product_log(image_product, base: float):
    """
    :param image_product: Image product to be modified
    :param base: base to which to take the log
    :return: Value of log base x of x - 1 + result of image product, repeated the number of times indicated
    """
    return lambda mainImg, tempImg: math.log(image_product(mainImg, tempImg) + base - 1, base)


def image_product_as_power(image_product, base: float):
    """
    :param image_product: Image product to be modified
    :param base: Base to which will be raised by (image product - 1)
    :return: Base raised by (image product - 1)
    """
    return lambda mainImg, tempImg: base ** (image_product(mainImg, tempImg) - 1)


def multiply_image_product(image_product, factor: float):
    """
    :param image_product: Image product to be modified
    :param factor: Factor to which to multiply the image product
    :return: Image product multiplied by factor
    """
    return lambda mainImg, tempImg: image_product(mainImg, tempImg) * factor


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
    size = len(imageSet)
    imageProductMatrix = np.ones((size, size)) * 2
    for index1, image1 in enumerate(imageSet):
        for index2, image2 in enumerate(imageSet):
            if index1 == index2:
                imageProductMatrix[index1][index2] = 1
            elif imageProductMatrix[index1][index2] < 2:
                continue
            else:
                result = imageProduct(image1, image2)
                imageProductMatrix[index1][index2] = result
                imageProductMatrix[index2][index1] = result
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

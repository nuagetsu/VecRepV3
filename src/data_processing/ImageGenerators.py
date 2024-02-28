import itertools
import re
import numpy as np
from numpy.typing import NDArray


def get_binary_image_set(imageLength: int, maxOnesPercentage=100) -> NDArray[int]:
    if maxOnesPercentage > 100:
        raise ValueError(str(maxOnesPercentage) + " > 100. Maximum percentage of ones has to be less than 100")
    cutoff = maxOnesPercentage * (imageLength ** 2) // 100
    fullList = []
    for item in itertools.product([0, 1], repeat=imageLength ** 2):
        if np.sum(item) <= cutoff:
            fullList.append(np.reshape(item, (imageLength, imageLength)))
    fullList = np.array(fullList)
    return fullList


def get_image_set(imageType: str):
    if re.search('[0-9]?[0-9]bin[0-9]?[0-9]max_ones$', imageType) is not None:  # Searching if image type follows the
        # format of 3bin40max_ones
        imageLength = int(re.search(r'^\d+', imageType).group())
        maxOnesPercentage = int(re.search(r'\d+', imageType[2:]).group())
        image_set = get_binary_image_set(imageLength, maxOnesPercentage=maxOnesPercentage)
    elif re.search('[0-9]?[0-9]bin$', imageType) is not None:  # Searching if the image type follows the format 2bin
        imageLength = int(re.search(r'\d+', imageType).group())
        image_set = get_binary_image_set(imageLength)
    else:
        raise ValueError(imageType + " is not a valid image type")
    return image_set

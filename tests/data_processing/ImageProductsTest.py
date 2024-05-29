import unittest
from typing import Callable

import numpy as np
import cv2
from src.data_processing.ImageProducts import (ncc, ncc_squared, ncc_pow, get_image_product,
                                               calculate_image_product_vector)

class TestNCC(unittest.TestCase):

    def test_get_image_product_vector(self):
        mainImgs = np.array([[[1, 0], [0, 0]], [[0, 1], [0, 0]], [[1, 0], [0, 1]], [[1, 1], [0, 1]]])
        tempImg = np.array([[1, 0], [0, 1]])
        correctResult = np.array([0.7071067, 0.7071067, 1, 0.816496])
        result = calculate_image_product_vector(tempImg, mainImgs, ncc)
        self.assertTrue(np.allclose(correctResult, result))

    def test_get_image_product(self):
        result = get_image_product("ncc")
        self.assertIsInstance(result, Callable)
        with self.assertRaises(ValueError):
            result = get_image_product("bcc")

    def test_ncc_with_zero_images(self):
        # Test when both mainImg and tempImg are zero arrays
        result = ncc(np.zeros((5, 5)), np.zeros((5, 5)))
        self.assertEqual(result, 1)

    def test_ncc_with_non_zero_images(self):
        # Test with non-zero mainImg and tempImg
        mainImg = np.random.rand(5, 5)
        tempImg = np.random.rand(5, 5)
        result = ncc(mainImg, tempImg)
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0)
        self.assertLessEqual(result, 1)


        # Test with translation
        mainImg = np.array([[0, 1], [1, 0]])
        tempImg = np.array([[1, 0], [0, 1]])
        result = ncc(mainImg, tempImg)
        self.assertAlmostEqual(result, 1, places=5)


        # Test with specific case
        mainImg = np.array([[0, 1], [1, 1]])
        tempImg = np.array([[1, 0], [0, 0]])
        result = ncc(mainImg, tempImg)
        self.assertAlmostEqual(result, 0.57735, places=5)

    def test_get_power_image_products(self):
        mainImg = np.array([[0, 1], [1, 1]])
        tempImg = np.array([[1, 0], [1, 0]])

        #test getting more image products that involve raising the power of ncc
        result1 = get_image_product("ncc_squared")
        self.assertIsInstance(result1, Callable)
        self.assertEqual(result1, ncc_squared)

        result2 = get_image_product("ncc_pow_2")
        self.assertIsInstance(result2, Callable)
        self.assertEqual(result2(mainImg, tempImg), ncc_pow(2)(mainImg, tempImg))

        result3 = get_image_product("ncc_pow_2.")
        self.assertIsInstance(result3, Callable)
        self.assertEqual(result3(mainImg, tempImg), ncc_pow(2)(mainImg, tempImg))

        result4 = get_image_product("ncc_pow_1.5")
        self.assertIsInstance(result4, Callable)
        self.assertEqual(result4(mainImg, tempImg), ncc_pow(1.5)(mainImg, tempImg))

if __name__ == '__main__':
    unittest.main()

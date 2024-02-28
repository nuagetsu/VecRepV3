import unittest
import numpy as np
import cv2
from src.data_processing.ImageProducts import ncc, get_image_product

class TestNCC(unittest.TestCase):

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


if __name__ == '__main__':
    unittest.main()

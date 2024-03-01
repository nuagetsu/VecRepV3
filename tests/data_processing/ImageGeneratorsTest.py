import unittest

import numpy as np

from src.data_processing.ImageGenerators import get_image_set, get_binary_image_set

class TestImageGenerators(unittest.TestCase):
    def test_output_shape(self):
        # Test the output shape for different imageLength values
        result_3x3 = get_binary_image_set(imageLength=3)
        self.assertIsInstance(result_3x3, np.ndarray)
        self.assertEqual(result_3x3.shape, (512, 3, 3))

    def test_maxOnesPercentage(self):
        # Test with maxOnesPercentage specified
        result_2x2_with_percentage = get_binary_image_set(imageLength=2, maxOnesPercentage=50)
        self.assertIsInstance(result_2x2_with_percentage, np.ndarray)
        self.assertEqual(result_2x2_with_percentage.shape, (11, 2, 2))

    def test_get_image_set_valid_input(self):
        results_2x2 = get_image_set("2bin")
        self.assertEqual(results_2x2.shape, (16, 2, 2))
        results_2x2_with_percentage = get_image_set("2bin50max_ones")
        self.assertEqual(results_2x2_with_percentage.shape, (11,2,2))

    def test_get_image_set_invalid_input(self):
        # Test invalid input format
        invalid_inputs = ["invalid_type", "2binzz", "2binmax_ones"]
        for imageType in invalid_inputs:
            with self.subTest(imageType=imageType):
                with self.assertRaises(ValueError):
                    get_image_set(imageType)
if __name__ == '__main__':
    unittest.main()
import unittest

import numpy as np
from src.data_processing import Filters


class TestFilters(unittest.TestCase):
    def test_get_trans_unique(self):
        # Define a sample input array of images
        images = np.array([[[1, 1], [0, 0]], [[0, 1], [0, 1]], [[1, 0], [1, 0]]])

        # Call the function with the sample input
        result = Filters.apply_translationally_unique_filter(images)

        # Check if the function returns the correct images
        self.assertEqual(len(result), 2)

    def test_get_only_one_island(self):
        # Define a sample input array of images
        images = np.array([[[1, 0], [0, 1]], [[1, 1], [0, 0]]])

        # Call the function with the sample input
        result = Filters.apply_one_island_filter(images)

        # Check if the function returns the correct images
        self.assertEqual(len(result), 1)
        self.assertTrue(np.array_equal(result[0], images[1]))

    def test_get_only_percent_size(self):
        # Define a sample input array of squares
        images = np.array([[[1, 0], [0, 0]], [[1, 1], [0, 0]]])

        # Call the function with the sample input
        result = Filters.apply_max_ones_filter(images, 45)

        # Check if the function returns the correct squares
        self.assertEqual(len(result), 1)
        self.assertTrue(np.array_equal(result[0], images[0]))

    def test_get_filtered_squares(self):
        # Define a sample input square and filters
        images = np.array([[[1, 0], [0, 0]], [[1, 0], [0, 1]], [[0, 1], [0, 0]]])
        filters = ["unique", "one_island"]

        # Call the function with the sample input
        result = Filters.get_filtered_image_sets(imageSet=images, filters=filters)

        # Check if the function returns the correct squares
        expected_result = Filters.apply_translationally_unique_filter(images)
        expected_result = Filters.apply_one_island_filter(images)
        for filter in filters:
            expected_result = Filters.apply_filter(expected_result, filter)
        self.assertTrue(np.array_equal(result, expected_result))


if __name__ == '__main__':
    unittest.main()

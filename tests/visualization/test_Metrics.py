from unittest import TestCase

import numpy as np
from src.visualization import Metrics

class Test(TestCase):
    def test_k_neighbour_score_simple(self):
        # test simple cases with no repeated values
        image_product_vec = np.array([1, 0.9, 0.8, 0.7, 0.6, 0.5])
        dot_product_vec = np.array([1, 0.8, 0.7, 0.6, 0.5, 0.4])
        ideal_result = 1
        score = Metrics.k_neighbour_score(image_product_vec, dot_product_vec, 3)
        self.assertAlmostEqual(score, ideal_result)

        image_product_vec = np.array([1, 0.9, 0.8, 0.7, 0.6, 0.5])
        dot_product_vec = np.array([1, 0.8, 0.7, 0.6, 0.5, 0.9])
        ideal_result = 2 / 3
        score = Metrics.k_neighbour_score(image_product_vec, dot_product_vec, 3)
        self.assertAlmostEqual(score, ideal_result)

        image_product_vec = np.array([1, 0.9, 0.8, 0.7, 0.6, 0.5])
        dot_product_vec = np.array([0.3, 0.8, 0.4, 0.6, 0.5, 0.9])
        ideal_result = 1 / 3
        score = Metrics.k_neighbour_score(image_product_vec, dot_product_vec, 3)
        self.assertAlmostEqual(score, ideal_result)


    def test_k_neighbour_score_repeated(self):
        # test cases with repeated values
        image_product_vec = np.array([1, 0.9, 0.9, 0.7, 0.7, 0.5])
        dot_product_vec = np.array([1, 0.8, 0.7, 0.6, 0.5, 0.4])
        ideal_result = 1
        score = Metrics.k_neighbour_score(image_product_vec, dot_product_vec, 2)
        self.assertAlmostEqual(score, ideal_result)

        score = Metrics.k_neighbour_score(image_product_vec, dot_product_vec, 4)
        self.assertAlmostEqual(score, ideal_result)

        image_product_vec = np.array([1, 0.9, 0.7, 0.7, 0.7, 0.5])
        dot_product_vec = np.array([0.8, 0.8, 0.7, 0.6, 0.5, 0.8])
        ideal_result = 2/3
        score = Metrics.k_neighbour_score(image_product_vec, dot_product_vec, 3)
        self.assertAlmostEqual(score, ideal_result)

        image_product_vec = np.array([1, 0.7, 0.7, 0.7, 0.7, 0.3])
        dot_product_vec = np.array([1, 0.5, 0.7, 0.5, 0.6, 0.3])
        ideal_result = 1
        score = Metrics.k_neighbour_score(image_product_vec, dot_product_vec, 3)
        self.assertAlmostEqual(score, ideal_result)

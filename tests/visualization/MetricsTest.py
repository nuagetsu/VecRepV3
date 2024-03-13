from unittest import TestCase

import numpy as np
from src.visualization import Metrics

class Test(TestCase):
    def test_get_k_neighbour_score_simple(self):
        # test simple cases with no repeated values
        image_product_vec = np.array([1, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5])
        dot_product_vec = np.array([1, 0.95, 0.8, 0.7, 0.6, 0.5, 0.4])
        ideal_result = 1
        score = Metrics.get_k_neighbour_score(image_product_vec, dot_product_vec, 3)
        self.assertAlmostEqual(score, ideal_result)

        image_product_vec = np.array([1, 0.95,  0.9, 0.8, 0.7, 0.6, 0.5])
        dot_product_vec = np.array([1, 0.95, 0.8, 0.7, 0.6, 0.5, 0.9])
        ideal_result = 2 / 3
        score = Metrics.get_k_neighbour_score(image_product_vec, dot_product_vec, 3)
        self.assertAlmostEqual(score, ideal_result)

        image_product_vec = np.array([1, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5])
        dot_product_vec = np.array([1, 0.3, 0.8, 0.4, 0.6, 0.5, 0.9])
        ideal_result = 1 / 3
        score = Metrics.get_k_neighbour_score(image_product_vec, dot_product_vec, 3)
        self.assertAlmostEqual(score, ideal_result)


    def test_get_k_neighbour_score_repeated(self):
        # test cases with repeated values
        image_product_vec = np.array([1, 0.95, 0.9, 0.9, 0.7, 0.7, 0.5])
        dot_product_vec = np.array([1, 0.95, 0.8, 0.7, 0.6, 0.5, 0.4])
        ideal_result = 1
        score = Metrics.get_k_neighbour_score(image_product_vec, dot_product_vec, 2)
        self.assertAlmostEqual(score, ideal_result)

        score = Metrics.get_k_neighbour_score(image_product_vec, dot_product_vec, 4)
        self.assertAlmostEqual(score, ideal_result)

        image_product_vec = np.array([1, 0.95, 0.9, 0.7, 0.7, 0.7, 0.5])
        dot_product_vec = np.array([1, 0.8, 0.8, 0.7, 0.6, 0.5, 0.8])
        ideal_result = 2/3
        score = Metrics.get_k_neighbour_score(image_product_vec, dot_product_vec, 3)
        self.assertAlmostEqual(score, ideal_result)

        image_product_vec = np.array([1, 0.95, 0.7, 0.7, 0.7, 0.7, 0.3])
        dot_product_vec = np.array([1, 0.95, 0.5, 0.7, 0.5, 0.6, 0.3])
        ideal_result = 1
        score = Metrics.get_k_neighbour_score(image_product_vec, dot_product_vec, 3)
        self.assertAlmostEqual(score, ideal_result)

    def test_frob_norm(self):

        m1 = np.array([[1, 1, 1, 0],
                      [1, 1, 1, 1],
                      [1, 1, 1, 1],
                      [0, 1, 1, 1]])


        m2 = np.array([[0.5, 1, 1, 0.25],
                      [1, 1, 1, 1],
                      [1, 1, 1, 1],
                      [0, 1, 1, 1]])

        frob_norm = Metrics.get_frob_distance(m1, m2)
        self.assertAlmostEqual(frob_norm, 0.55901699)

    def test_get_plotting_data(self):

        G = np.array([[1, 0.25, 0.5, 0.5],
                      [0.25, 1, 0.33, 0.7],
                      [0.5, 0.33, 1, 0.5],
                      [0.5, 0.7, 0.5, 1]])
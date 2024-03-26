from unittest import TestCase

import numpy as np
from src.visualization import Metrics


class Test(TestCase):
    def test_get_k_neighbour_score_simple(self):
        # test simple cases with no repeated values
        image_product_vec = np.array([1, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5])
        dot_product_vec = np.array([1, 0.95, 0.8, 0.7, 0.6, 0.5, 0.4])
        ideal_result = 3
        score = Metrics.get_k_neighbour_score(image_product_vec, dot_product_vec, 3)
        self.assertAlmostEqual(score, ideal_result)

        image_product_vec = np.array([1, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5])
        dot_product_vec = np.array([1, 0.95, 0.8, 0.7, 0.6, 0.5, 0.9])
        ideal_result = 2
        score = Metrics.get_k_neighbour_score(image_product_vec, dot_product_vec, 3)
        self.assertAlmostEqual(score, ideal_result)

        image_product_vec = np.array([1, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5])
        dot_product_vec = np.array([1, 0.3, 0.8, 0.4, 0.6, 0.5, 0.9])
        ideal_result = 1
        score = Metrics.get_k_neighbour_score(image_product_vec, dot_product_vec, 3)
        self.assertAlmostEqual(score, ideal_result)

    def test_get_k_neighbour_score_repeated(self):
        # test cases with repeated values
        image_product_vec = np.array([1, 0.95, 0.9, 0.9, 0.7, 0.7, 0.5])
        dot_product_vec = np.array([1, 0.95, 0.8, 0.7, 0.6, 0.5, 0.4])
        ideal_result = 2
        score = Metrics.get_k_neighbour_score(image_product_vec, dot_product_vec, 2)
        self.assertAlmostEqual(score, ideal_result)

        score = Metrics.get_k_neighbour_score(image_product_vec, dot_product_vec, 4)
        self.assertAlmostEqual(score, 4)

        image_product_vec = np.array([1, 0.95, 0.9, 0.7, 0.7, 0.7, 0.5])
        dot_product_vec = np.array([1, 0.8, 0.8, 0.7, 0.6, 0.5, 0.8])
        ideal_result = 2
        score = Metrics.get_k_neighbour_score(image_product_vec, dot_product_vec, 3)
        self.assertAlmostEqual(score, ideal_result)

        image_product_vec = np.array([1, 0.95, 0.7, 0.7, 0.7, 0.7, 0.3])
        dot_product_vec = np.array([1, 0.95, 0.5, 0.7, 0.5, 0.6, 0.3])
        ideal_result = 3
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

    def test_mean_k_neighbour_score(self):
        G = np.array([[1, 0.25, 0.5, 0.5],
                      [0.25, 1, 0.33, 0.7],
                      [0.5, 0.33, 1, 0.5],
                      [0.5, 0.7, 0.5, 1]])
        Gprime = np.array([[1, 0.25, 0.5, 0.5],
                      [0.25, 1, 0.33, 0.7],
                      [0.5, 0.33, 1, 0.5],
                      [0.5, 0.7, 0.5, 1]])
        k_neigh_scores1 = Metrics.get_mean_normed_k_neighbour_score(G, Gprime, 1)
        k_neigh_scores2 = Metrics.get_mean_normed_k_neighbour_score(G, Gprime, 2)
        k_neigh_scores3 = Metrics.get_mean_normed_k_neighbour_score(G, Gprime, 3)
        self.assertEqual(k_neigh_scores1, 1)
        self.assertEqual(k_neigh_scores2, 1)
        self.assertEqual(k_neigh_scores3, 1)


        G = np.array([[1, 0.25, 0.5, 0.5],
                      [0.25, 1, 0.33, 0.7],
                      [0.5, 0.33, 1, 0.5],
                      [0.8, 0.7, 0.5, 1]])
        Gprime = np.array([[1, 0.25, 0.5, 0.5],
                      [0.5, 1, 0.33, 0.7],
                      [0.5, 0.33, 1, 0.5],
                      [0.6, 0.7, 0.5, 1]])
        k_neigh_scores1 = Metrics.get_mean_normed_k_neighbour_score(G, Gprime, 1)
        k_neigh_scores2 = Metrics.get_mean_normed_k_neighbour_score(G, Gprime, 2)
        k_neigh_scores3 = Metrics.get_mean_normed_k_neighbour_score(G, Gprime, 3)
        self.assertAlmostEqual(k_neigh_scores1, 0.75)
        self.assertAlmostEqual(k_neigh_scores2, 0.875)
        self.assertAlmostEqual(k_neigh_scores3, 1)



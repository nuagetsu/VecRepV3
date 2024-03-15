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

    def test_apply_k_neighbour_score(self):
        G = np.array([[1, 0.25, 0.5, 0.5],
                      [0.25, 1, 0.33, 0.7],
                      [0.5, 0.33, 1, 0.5],
                      [0.5, 0.7, 0.5, 1]])
        A = np.array([[1, 0.25, 0.5, 0.5],
                      [0.25, 1, 0.33, 0.7],
                      [0.5, 0.33, 1, 0.5],
                      [0.5, 0.7, 0.5, 1]])
        k_neigh_scores = Metrics.apply_k_neighbour(G, A, 1, 3)
        correctResult = [{"kval": 1, "neighbourScore": [1, 1, 1, 1]}, {"kval": 2, "neighbourScore": [1, 1, 1, 1]},
                         {"kval": 3, "neighbourScore": [1, 1, 1, 1]}]
        self.assertTrue(correctResult == k_neigh_scores)


        G = np.array([[1, 0.25, 0.5, 0.5],
                      [0.25, 1, 0.33, 0.7],
                      [0.5, 0.33, 1, 0.5],
                      [0.8, 0.7, 0.5, 1]])
        A = np.array([[1, 0.25, 0.5, 0.5],
                      [0.5, 1, 0.33, 0.7],
                      [0.5, 0.33, 1, 0.5],
                      [0.6, 0.7, 0.5, 1]])
        k_neigh_scores = Metrics.apply_k_neighbour(G, A, 1, 3)
        correctResult = [{"kval": 1, "neighbourScore": [1, 1, 1, 0]}, {"kval": 2, "neighbourScore": [1, 0.5, 1, 1]},
                         {"kval": 3, "neighbourScore": [1, 1, 1, 1]}]
        self.assertTrue(correctResult == k_neigh_scores)



    def test_get_plotting_data(self):
        G = np.array([[1, 0.25, 0.5, 0.5],
                      [0.25, 1, 0.33, 0.7],
                      [0.5, 0.33, 1, 0.5],
                      [0.5, 0.7, 0.5, 1]])
        A = np.array([[-7.13082409e-01, -7.44011971e-01, -7.44312002e-01, -8.88748870e-01],
                      [-5.26566311e-01, 6.02700095e-01, -3.89545229e-01, 2.44177275e-01],
                      [4.44912115e-01, -3.92523115e-17, -5.41109329e-01, 9.61972140e-02],
                      [-1.27650337e-01, -2.88441992e-01, -3.81425294e-02, 3.75831079e-01]])
        plottingData = Metrics.get_plotting_data(G, A)
        self.assertAlmostEqual(plottingData.frobDistance, 0)
        self.assertTrue(np.allclose(plottingData.initialEigenvalues, np.array([2.40591524, 0.85188751, 0.5, 0.24219724])))
        self.assertTrue(np.allclose(plottingData.finalEigenvalues, np.array([2.40591524, 0.85188751, 0.5, 0.24219724])))
        correctResult = [{"kval": 1, "neighbourScore": [1, 1, 1, 1]}, {"kval": 2, "neighbourScore": [1, 1, 1, 1]}]
        self.assertTrue(plottingData.kNeighbourScores == correctResult)
        self.assertAlmostEqual(plottingData.aveFrobDistance, 0)

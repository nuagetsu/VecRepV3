import unittest
import numpy as np
from src.data_processing import EmbeddingFunctions


class TestEmbeddingFunction(unittest.TestCase):
    def test_emb_similarity(self):
        # Test for a basic rotation
        matrix1 = np.array([[1, 1, 1],
                            [1, 1, 0],
                            [1, 0, 0]])

        rotMat = np.array([[0.5449850, -0.2201882, 0.8090170],
                           [0.7151808, 0.6256976, -0.3114787],
                           [-0.4376160, 0.7483447, 0.4984702]])
        matrix2 = np.matmul(rotMat, matrix1)
        self.assertTrue(EmbeddingFunctions.check_emb_similarity(matrix1, matrix2))

        # Test for a random transformation
        matrix1 = np.array([[1, 1, 1],
                            [1, 1, 0],
                            [1, 0, 0]])
        randomMat = np.array([[0.5449850, -0.2201882, 0.8090170],
                              [0.7151808, 0.6256976, -0.3114787],
                              [1, 1, 0]])
        matrix2 = np.matmul(randomMat, matrix1)
        self.assertFalse(EmbeddingFunctions.check_emb_similarity(matrix1, matrix2))

    def test_eig_for_symmetric(self):
        # Test non symmetric matrix
        G = np.array([[1, 1, 1],
                      [1, 0, 0],
                      [0, 0, 0]])

        with self.assertRaises(ValueError):
            eig, vec = EmbeddingFunctions.get_eig_for_symmetric(G)



        # Define a sample input matrix
        G = np.array([[1, 0.25, 0.5, 0.5],
                      [0.25, 1, 0.33, 0.7],
                      [0.5, 0.33, 1, 0.5],
                      [0.5, 0.7, 0.5, 1]])

        # Call the function with the sample input
        eig, vec = EmbeddingFunctions.get_eig_for_symmetric(G)

        resultEig = np.array([2.406, 0.852, 0.5, 0.242])

        # Check if the dot product of A and A transpose equals B
        self.assertTrue(np.allclose(eig, resultEig, atol=1e-3))

    def test_matrix_decomposition(self):
        # Define a sample input matrix
        G = np.array([[1, 0.25, 0.5, 0.5],
                      [0.25, 1, 0.33, 0.7],
                      [0.5, 0.33, 1, 0.5],
                      [0.5, 0.7, 0.5, 1]])

        # Call the function with the sample input
        A = EmbeddingFunctions.get_embeddings_zeroed(G)
        Gprime = np.matmul(A.T, A)
        # Check if the dot product of A transpose and A equals G
        self.assertTrue(np.allclose(Gprime, G))

        # Test for non positive semidefinite matrix
        G = np.array([[1, 1, 1, 0],
                      [1, 1, 1, 1],
                      [1, 1, 1, 1],
                      [0, 1, 1, 1]])
        with self.assertRaises(EmbeddingFunctions.NonPositiveSemidefiniteError):
            A = EmbeddingFunctions.get_embeddings_zeroed(G)

    def test_neg_zeroed(self):
        # Test positive semidef input
        G = np.array([[4, 12, -16], [12, 37, -43], [-16, -43, 98]])

        # Call the function with the sample input
        A = EmbeddingFunctions.get_embeddings_negatives_zeroed(G)

        # Check if the dot product of A transpose and A equals G
        self.assertTrue(np.allclose(np.dot(A.T, A), G))

        # Test Non pos semidef input
        G = np.array([[1, 1, 1, 0], [1, 1, 1, 1], [1, 1, 1, 1], [0, 1, 1, 1]])

        A = EmbeddingFunctions.get_embeddings_negatives_zeroed(G)
        result = np.array([[1.17437573, 0.86387531, 0.86387531, 0.17437573],
                           [0.86387531, 1.10662427, 1.10662427, 0.86387531],
                           [0.86387531, 1.10662427, 1.10662427, 0.86387531],
                           [0.17437573, 0.86387531, 0.86387531, 1.17437573]])
        Gprime = np.matmul(A.T, A)
        self.assertTrue(np.allclose(Gprime, result, rtol=1e-2))

    def test_nDim_zeroed(self):
        # Test nDim error cases
        G = np.array([[4, 12, -16], [12, 37, -43], [-16, -43, 98]])

        with self.assertRaises(ValueError):
            A = EmbeddingFunctions.get_embeddings_zeroed(G, 9)

        with self.assertRaises(ValueError):
            A = EmbeddingFunctions.get_embeddings_zeroed(G, 0)

        # Test positive semidef input
        G = np.array([[4, 12, -16], [12, 37, -43], [-16, -43, 98]])

        # Call the function with the sample input
        A = EmbeddingFunctions.get_embeddings_zeroed(G, 3)
        Gprime = np.matmul(A.T, A)

        # Check if the dot product of A transpose and A equals G
        self.assertTrue(np.allclose(Gprime, G))

        # Test Non pos semidef input, nDim = 2
        G = np.array([[1, 1, 1, 0], [1, 1, 1, 1], [1, 1, 1, 1], [0, 1, 1, 1]])

        A = EmbeddingFunctions.get_embeddings_zeroed(G, 2)
        result = np.array([[1.17437573, 0.86387531, 0.86387531, 0.17437573],
                           [0.86387531, 1.10662427, 1.10662427, 0.86387531],
                           [0.86387531, 1.10662427, 1.10662427, 0.86387531],
                           [0.17437573, 0.86387531, 0.86387531, 1.17437573]])
        Gprime = np.matmul(A.T, A)
        self.assertTrue(np.allclose(Gprime, result, rtol=1e-2))

        # Test ndim = 1
        G = np.array([[1, 1, 1, 0], [1, 1, 1, 1], [1, 1, 1, 1], [0, 1, 1, 1]])

        A = EmbeddingFunctions.get_embeddings_zeroed(G, 1)
        result = np.array([[0.67437573, 0.86387531, 0.86387531, 0.67437573],
                           [0.86387531, 1.10662427, 1.10662427, 0.86387531],
                           [0.86387531, 1.10662427, 1.10662427, 0.86387531],
                           [0.67437573, 0.86387531, 0.86387531, 0.67437573]])
        Gprime = np.matmul(A.T, A)
        self.assertTrue(np.allclose(Gprime, result, rtol=1e-2))

    def test_matrix_decomposition_NC(self):# Define a sample input matrix
        G = np.array([[1, 0.25, 0.5, 0.5], [0.25, 1, 0.33, 0.7], [0.5, 0.33, 1, 0.5], [0.5, 0.7, 0.5, 1]])

        # Call the function with the sample input
        A = EmbeddingFunctions.get_embeddings_nc(G)
        Gprime = np.matmul(A.T, A)
        # Check if the dot product of A transpose and A equals G
        self.assertTrue(np.allclose(Gprime, G))

        G = np.array([[2, -1, 0, 0],
                      [-1, 2, -1, 0],
                      [0, -1, 2, -1],
                      [0, 0, -1, 2]])

        # Call the function with the sample input
        A = EmbeddingFunctions.get_embeddings_nc(G, nDim=3)
        Gprime = np.array([[1., -0.8084125, 0.1915875, 0.10677505],
                          [-0.8084125, 1., -0.65623269, 0.1915875],
                          [0.1915875, -0.65623269, 1., -0.8084125],
                          [0.10677505, 0.1915875, -0.8084125, 1.]])
        Ares = np.dot(A.T, A)
        # Check if the mat mul of At and A transpose equals G
        self.assertTrue(np.allclose(Gprime, Ares, atol=1e-4))
        self.assertEqual(A.shape, (3, 4))



if __name__ == '__main__':
    unittest.main()

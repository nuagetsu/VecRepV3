import math
import re

import numpy as np
from numpy.typing import NDArray
from sklearn.preprocessing import normalize

from src.helpers.NearestCorrelation import nearcorr, ExceededMaxIterationsError


class NonPositiveSemidefiniteError(Exception):
    pass


def get_embeddings_negatives_zeroed(matrixG):
    """
    :param matrixG: MatrixG to decompose
    :return: embeddings generated after all negative eigenvalue of the image product matrix are zeroed
    """
    eigenvalues, eigenvectors = get_eig_for_symmetric(matrixG)
    eigenvalues[0 > eigenvalues] = 0
    Droot = np.sqrt(np.diag(eigenvalues))

    matrixA = np.matmul(Droot, eigenvectors.T)
    return matrixA


def get_embeddings_mikecroucher_nc(matrixG, nDim=None) -> NDArray:
    """
    :param matrixG: Matrix to be decomposed
    :param nDim: Number of dimensions of vector embeddings in the embedding matrix
    :return: An embedding matrix with dimensions: nDim by len(matrix G)
    Approximates matrix G to a Positive semi-definite matrix using the nearest corrolation algorithm by mike croucher
    """
    # make the matrix symmetric
    if not check_symmetric(matrixG):
        raise ValueError("Matrix G has to be a symmetric matrix")
    matrixG = (matrixG + matrixG.T) / 2
    # Use the NC matrix algo
    try:
        matrixGprime = nearcorr(matrixG, max_iterations=10000)
    except ExceededMaxIterationsError:
        print("No NC matrix found after 10000 iterations")
        raise
    # Decompose the matrix
    if nDim is None:
        nDim = len(matrixG)
    matrixA = get_embeddings_mPCA(matrixGprime, nDim)
    return matrixA

def get_embeddings_matlab_nc(matrixG: NDArray, nDim: int) -> NDArray:
    # make the matrix symmetric
    if not check_symmetric(matrixG):
        raise ValueError("Matrix G has to be a symmetric matrix")
    matrixG = (matrixG + matrixG.T) / 2
    # Use the NC matrix algo


def get_embedding_matrix(imageProductMatrix: NDArray, embeddingType: str, nDim=None):
    """
    :param imageProductMatrix: Image product matrix to generate vectors
    :param embeddingType: Type of method to generate vector embeddings
    :param nDim: Number of dimensions in the vector embeddings.
    If none that means the nDim = length of image product matrix
    :return:
    """

    if embeddingType == "zero_neg":
        embeddingMatrix = get_embeddings_negatives_zeroed(imageProductMatrix)

    elif re.search('zero_[0-9]?[0-9]$', embeddingType) is not None:
        nDim = int(re.search(r'\d+', embeddingType).group())
        embeddingMatrix = get_embeddings_mPCA(imageProductMatrix, nDim)
    elif embeddingType == "nc":
        embeddingMatrix = get_embeddings_mikecroucher_nc(imageProductMatrix)
    elif re.search('nc_[0-9]?[0-9]$', embeddingType) is not None:
        nDim = int(re.search(r'\d+', embeddingType).group())
        embeddingMatrix = get_embeddings_mikecroucher_nc(imageProductMatrix, nDim=nDim)
    else:
        raise ValueError(embeddingType + " is not a valid embedding type")
    return embeddingMatrix


def get_eig_for_symmetric(matrixG: NDArray) -> (NDArray, NDArray):
    """
    :param matrixG: Symmetric matrix G
    :return: Eigenvalues, transposed eigenvectors for matrix G.

    Checks that the matrix is symmetric then returns the sorted (in descending order) eigenvalues and the transposed eigenvectors
    """
    if not check_symmetric(matrixG):
        raise ValueError("Matrix G has to be a symmetric matrix")
    # The eigenvectors are already given in the form of a transposed eigenvector matrix, where the rows represent the
    # eigenvectors instead of columns
    eigenvalues, eigenvectors = np.linalg.eigh(matrixG)
    # Reverses the order form ascending to descending
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors.T[::-1].T
    return eigenvalues, eigenvectors



def get_embeddings_mPCA(matrixG: NDArray, nDim=None):
    """
    :param matrixG: Matrix to be decomposed
    :param nDim: Number of dimensions of the vector embedding. If none, then carries out a normal decomposition
    :return: An embedding matrix, with each vector having nDim dimensions.
    Keeps the largest nDim number of eigenvalues and zeros the rest. Followed by the normalization of the embedding matrix
    Effectively applies a modified PCA to the vector embeddings
    """
    maxDim = len(matrixG[0])
    if nDim is None:
        nDim = maxDim
    if nDim <= 0:
        raise ValueError(str(nDim) + " is <= 0. nDim has to be > 0")
    if nDim > maxDim:
        raise ValueError(str(nDim) + " is > " + str(maxDim) + ". nDim has to be < the length of matrixG")

    eigenvalues, eigenvectors = get_eig_for_symmetric(matrixG)

    # Checking that the matrix is positive semi-definite
    for eigenvalue in eigenvalues[:nDim]:
        if not (eigenvalue > 0 or math.isclose(eigenvalue, 0, abs_tol=1e-5)):
            raise NonPositiveSemidefiniteError("Even after zeroing smaller eigenvalues, matrix G is not a positive "
                                               "semi-definite matrix. Consider increasing the value of nDim")

    # Zeros negative eigenvalues which are close to zero
    eigenvalues[0 > eigenvalues] = 0
    Droot = np.sqrt(np.diag(eigenvalues))

    # Slices the diagonal matrix to remove smaller eigenvalues
    Drootm = Droot[:nDim, :]

    matrixA = np.matmul(Drootm, eigenvectors.T)

    # Normalizing matrix A
    matrixA = normalize(matrixA, norm='l2', axis=0)
    return matrixA


def check_symmetric(a, atol=1e-05):
    return np.allclose(a, a.T, atol=atol)


def check_emb_similarity(matrix1: NDArray, matrix2: NDArray) -> bool:
    """
    :param matrix1: Embedding matrix 1
    :param matrix2: Embedding matrix 2
    :return: If the 2 matrices are a simple rotation of each other (similar to 4dp)
    """
    if matrix1.shape != matrix2.shape:
        return False
    Q1, R1 = np.linalg.qr(matrix1)
    Q2, R2 = np.linalg.qr(matrix2)
    return np.allclose(R1, R2, atol=1e-4)

import math
import re

import numpy as np
from numpy.typing import NDArray
from oct2py import octave
from sklearn.preprocessing import normalize

from src.helpers.FilepathUtils import get_matlab_dirpath


class NonPositiveSemidefiniteError(Exception):
    pass


def pencorr(matrixG: NDArray, nDim: int) -> NDArray:
    """
    :param matrixG: Symmetric square matrix
    :param nDim: Number of non-zero eigenvalues in the output matrix
    :return: matrix G', a symmetric matrix with the same dimension as matrix G, which is the nearest correlation matrix with
    nDim non-zero eigenvalues

    Uses oct2py to run this matlab code by Prof SUN Defeng. The exact code used is slightly modified version of the
    files in Rank_CaliMat, which replaces the mexeig.c file with a mexeig.m file with similar functionality
    https://www.polyu.edu.hk/ama/profile/dfsun/Rank_CaliMatHdm.zip
    """
    matrixG, nDim = is_valid_matrix_g(matrixG, nDim)

    # TODO what if solver fails
    matlabDir = get_matlab_dirpath()
    _ = octave.addpath(matlabDir)
    octave.push("n", len(matrixG))
    octave.push("r_rank", nDim)
    octave.push("G", matrixG)

    # For fixed diagonal constraint
    octave.eval("I_e = [1:1:n]'; J_e = I_e; k_e = length(I_e);")

    # To generate the bound e,l & u
    octave.eval("e = ones(n,1);  ConstrA.e = e; ConstrA.Ie = I_e; ConstrA.Je = J_e;")

    # Set options
    octave.eval("OPTIONS.tau    = 0; OPTIONS.tolrel = 1.0e-5;")

    # Execute function
    octave.eval("[X,INFOS] = PenCorr(G,ConstrA,r_rank,OPTIONS);", verbose=False)
    matrixGprime = octave.pull("X")
    return matrixGprime

def pencorr_weighted(matrixG: NDArray, nDim: int, matrixH: NDArray):
    """
    :param matrixG: Symmetric square matrix
    :param nDim: Number of non-zero eigenvalues in the output matrix
    :param matrixH: Weight Matrix
    :return: matrix G', a symmetric matrix with the same dimension as matrix G, which is the nearest correlation matrix with
    nDim non-zero eigenvalues

    Weighted version of the above code included in the same link.
    """
    matrixG, nDim = is_valid_matrix_g(matrixG, nDim)

    matlabDir = get_matlab_dirpath()
    _ = octave.addpath(matlabDir)
    octave.push("n", len(matrixG))
    octave.push("r_rank", nDim)
    octave.push("G", matrixG)
    octave.push("H", matrixH)

    # For fixed diagonal constraint
    octave.eval("I_e = [1:1:n]'; J_e = I_e; k_e = length(I_e);")

    # To generate the bound e,l & u
    octave.eval("e = ones(n,1);  ConstrA.e = e; ConstrA.Ie = I_e; ConstrA.Je = J_e;")

    # Set options
    octave.eval("OPTIONS.tau    = 0; OPTIONS.tolrel = 1.0e-5;")

    # Execute function
    octave.eval("[X,INFOS] = PenCorr_HnormMajorDiag(G,H,ConstrA,r_rank,OPTIONS);", verbose=False)
    matrixGprime = octave.pull("X")
    return matrixGprime

def get_embedding_matrix(imageProductMatrix: NDArray, embeddingType: str, nDim=None):
    """
    :param imageProductMatrix: Image product matrix to generate vectors
    :param embeddingType: Type of method to generate vector embeddings
    :param nDim: Number of dimensions in the vector embeddings.
    If none that means the nDim = length of image product matrix
    :return:
    """
    if re.search('pencorr_[0-9]*[0-9]$', embeddingType) is not None:
        nDim = int(re.search(r'\d+', embeddingType).group())
        matrixGprime = pencorr(imageProductMatrix, nDim)
        embeddingMatrix = get_embeddings_mPCA(matrixGprime, nDim)
    elif re.search('pencorr_[0-9]*[0-9]_weight_[0-9]*[0-9]$', embeddingType) is not None:
        match = re.findall(r'\d+', embeddingType)
        nDim = int(match[0])
        weight = generate_weightings(imageProductMatrix, int(match[1]))
        matrixGprime = pencorr_weighted(imageProductMatrix, nDim, weight)
        embeddingMatrix = get_embeddings_mPCA(matrixGprime, nDim)
    else:
        raise ValueError(embeddingType + " is not a valid embedding type")
    return embeddingMatrix


def get_eig_for_symmetric(matrixG: NDArray) -> (NDArray, NDArray):
    """
    :param matrixG: Symmetric matrix G
    :return: Eigenvalues, eigenvectors for matrix G.

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
    :param matrixG: Matrix G to be decomposed
    :param nDim: Number of dimensions of the vector embedding.
    :return: An embedding matrix, with each vector having nDim dimensions. An nDim by len(MatrixG) matrix
    TLDR: Input matrix G' into the function, with the number of dimensions you want your vectors to have.
    Function will then output matrix A.

    1. Keeps the largest nDim number of eigenvalues and zeros the rest.
    2. Decomposes the matrix G into matrix A
    3. Normalize the embedding matrix

    """
    # Check and clean input
    matrixG, nDim = is_valid_matrix_g(matrixG, nDim)

    eigenvalues, eigenvectors = get_eig_for_symmetric(matrixG)

    # Checking that the matrix is positive semi-definite
    for eigenvalue in eigenvalues[:nDim]:
        if not (eigenvalue > 0 or math.isclose(eigenvalue, 0, abs_tol=1e-5)):
            raise NonPositiveSemidefiniteError("Matrix G does not have al least nDim number of positive eigenvalues")

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


def is_valid_matrix_g(matrixG: NDArray, nDim) -> (NDArray, int):
    """
    :param matrixG: Square symmetric matrix to check
    :param nDim: nDim value to check
    :return: Checks that matrix G is a symmetric, square matrix. Also checks that nDim
    is positive non-zero and less than the length of matrix G. Raises an error otherwise.
    If it is valid, return a symmetric matrix G and nDim
    """
    if not check_symmetric(matrixG):
        raise ValueError("Matrix G has to be a symmetric matrix")
    if not all(len(row) == len(matrixG) for row in matrixG):
        raise ValueError("Matrix G must be a square matrix")

    # make the matrix symmetric
    matrixG = (matrixG + matrixG.T) / 2

    maxDim = len(matrixG[0])
    if nDim is None:
        nDim = maxDim
    if nDim <= 0:
        raise ValueError(str(nDim) + " is <= 0. nDim has to be > 0")
    if nDim > maxDim:
        raise ValueError(str(nDim) + " is > " + str(maxDim) + ". nDim has to be < the length of matrixG")

    return matrixG, nDim

def generate_weightings(matrixG: NDArray, index: int, k=5) -> NDArray:
    """
    :param matrixG: Matrix G to be decomposed
    :return: Weightings through which to run weighted pencorr
    """
    if index == 0:
        return np.ones((len(matrixG), len(matrixG)))
    elif index == 1:
        return matrixG
    elif index == 2:
        return matrixG ** 2
    elif index == 3:
        return matrixG ** 3
    elif index == 4:
        nbr_arr = matrixG.transpose()
        k = k + 1
        weight_arr = np.zeros_like(matrixG)
        for row in range(len(nbr_arr)):
            max_index = np.argpartition(matrixG[row], -k)[-k:]
            kth_largest = nbr_arr[row][max_index[0]]
            kth_element = np.where(nbr_arr[row] == kth_largest)
            max_index = np.union1d(max_index, kth_element)
            for n in max_index:
                weight_arr[row][n] = nbr_arr[row][n]
        weight_arr = np.asarray(weight_arr)
        return weight_arr.transpose()
    elif index == 5:
        nbr_arr = matrixG.transpose()
        k = k + 5
        weight_arr = np.zeros_like(matrixG)
        for row in range(len(nbr_arr)):
            max_index = np.argpartition(matrixG[row], -k)[-k:]
            kth_largest = nbr_arr[row][max_index[0]]
            kth_element = np.where(nbr_arr[row] == kth_largest)
            max_index = np.union1d(max_index, kth_element)
            for n in max_index:
                weight_arr[row][n] = 1
        weight_arr = np.asarray(weight_arr)
        return weight_arr.transpose()
    elif index == 6:
        nbr_arr = matrixG.transpose()
        k = k + 5
        weight_arr = np.zeros_like(matrixG)
        for row in range(len(nbr_arr)):
            max_index = np.argpartition(matrixG[row], -k)[-k:]
            kth_largest = nbr_arr[row][max_index[0]]
            kth_element = np.where(nbr_arr[row] == kth_largest)
            max_index = np.union1d(max_index, kth_element)
            for n in max_index:
                weight_arr[row][n] = nbr_arr[row][n]
        weight_arr = np.asarray(weight_arr)
        return weight_arr.transpose()
    elif index == 7:
        return matrixG ** 5
    elif index == 9:        #Testing
        weight = matrixG
        for i in range(len(matrixG)):
            weight[i][i] = 0
        return weight
    elif index == 10:
        nbr_arr = matrixG.transpose()
        k = k + 1
        weight_arr = np.zeros_like(matrixG)
        for row in range(len(nbr_arr)):
            max_index = np.argpartition(matrixG[row], -k)[-k:]
            kth_largest = nbr_arr[row][max_index[0]]
            kth_element = np.where(nbr_arr[row] == kth_largest)
            max_index = np.union1d(max_index, kth_element)
            for n in max_index:
                weight_arr[row][n] = 1
        weight_arr = np.asarray(weight_arr)
        return weight_arr.transpose()
    else:
        raise ValueError(str(index) + "is not a valid weighting index")

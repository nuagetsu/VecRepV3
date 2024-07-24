import numpy as np
from scipy import optimize
from sympy import *

from src.data_processing.ImageProducts import get_image_product, calculate_image_product_vector


# The following method was written by Lim Cheng Ze Jed
def Lagrangian_Method2(A, b):
    """
    :INPUTS:
    A: The vector embedding matrix
    b: The image products of the selected image and the subset of images

    :OUTPUTS:
    final_dist: The vector representation matrix
    selected_lambda: The optimised lagragian multiplier

    """

    # Below is the math done to find the multiplier
    A = np.array(A)
    b = np.array(b)
    M = np.dot(A, A.T)
    S = np.diag(np.diag(M))
    eigenvalues = S
    S_len = len(S)
    y = np.dot(A, b)

    lamda = symbols('lamda')
    S = Matrix(S)
    S2 = (S + lamda * eye(S_len)) ** (-2)
    y = Matrix(y)
    H = y.T * S2 * y

    lst_of_eigenvalues = set(np.diag(eigenvalues))
    lst_of_eigenvalues = sorted(lst_of_eigenvalues)
    for j in range(len(lst_of_eigenvalues)):  # to find the roots in between the peaks
        lst_of_eigenvalues[j] = -lst_of_eigenvalues[j]

    # ADD THE MINUS ONE!!!
    equation = H[0] - 1  # use the sympy equation to find the absolute best lambda value
    f = lambdify(lamda, equation, 'numpy')

    initial_point_lst = []
    if len(lst_of_eigenvalues) == 1:
        initial_point_lst.append(lst_of_eigenvalues[0] - 100)  # negative
        initial_point_lst.append(lst_of_eigenvalues[0] - 0.1)  # positive
        initial_point_lst.append(lst_of_eigenvalues[0] + 100)  # negative
        initial_point_lst.append(lst_of_eigenvalues[-1] + 0.1)  # positive
    else:
        initial_point_lst.append(lst_of_eigenvalues[0] - 100)  # negative
        initial_point_lst.append(lst_of_eigenvalues[0] - 0.1)  # positive
        initial_point_lst.append(lst_of_eigenvalues[0] + 100)  # negative
        initial_point_lst.append(lst_of_eigenvalues[-1] + 0.1)  # positive
        for i in range(1, len(lst_of_eigenvalues)):
            mid_point = (lst_of_eigenvalues[i] + lst_of_eigenvalues[i - 1]) / 2
            left_of_eigenvalue = lst_of_eigenvalues[i] - 0.1
            right_of_eigenvalue = lst_of_eigenvalues[i - 1] + 0.1
            if f(mid_point) < 0:
                initial_point_lst.append(right_of_eigenvalue)
                initial_point_lst.append(mid_point)
                initial_point_lst.append(left_of_eigenvalue)
    z = set()  # solution set (lagrange multipliers)
    # sets different initial points throughout the entire graph
    for i in range(1, len(initial_point_lst)):
        if (f(initial_point_lst[i - 1]) < 0 and f(initial_point_lst[i]) > 0) or (f(initial_point_lst[i - 1]) > 0 and f(initial_point_lst[i]) < 0):
            solution = optimize.root_scalar(f, bracket=[initial_point_lst[i - 1], initial_point_lst[i]])
            z.add(solution.root)
    # Cleaning data for any -lambda values
    lagrangian_lst = []
    for ga in z:
        if str(ga) != "nan":
            lagrangian_lst.append(ga)

    # iterates through all the possible lagrangian values and output the one that provides the minimum ||Ax-b||
    final_dist = float('inf')
    selected_lambda = x_final = 0
    y = np.dot(A, b)

    for lamda in lagrangian_lst:
        x = np.dot((S + lamda * eye(S_len)) ** (-1), y)
        x = x.astype(np.float32)

        mat = np.dot(A.T, x) - b
        dist = np.dot(mat, mat)
        if dist < final_dist and 0.99 <= np.dot(x, x) <= 1.01:
            x_final = x
            final_dist = dist
            selected_lambda = lamda

    return x_final, final_dist, selected_lambda


def alt_lagrangian_method(A, b):
    A = np.array(A)
    b = np.array(b)
    M = np.dot(A, A.T)
    D = np.diag(M)
    eigenvalues = D.copy()
    D_len = len(D)
    y = np.dot(A, b)
    sorted(eigenvalues, reverse=True)

    def f(x):
        """
        Equation to solve. Must take in a numpy array and output a numpy array.
        :param x: Input variable lambda.
        :return: Output variable, to be equal to 0.
        """
        total = 0
        x = x[0]
        for index, eigenvalue in enumerate(D):
            term = (y[index] / (eigenvalue + x)) ** 2
            total += term
        return np.array([total - 1])

    # Use Scipy to code min max search etc. Or root find





def get_embedding_estimate(image_input, training_image_set, image_product: str, embedding_matrix):
    """
    :param image_input: Image of the same dimensions as that used in the training image set
    :param training_image_set: Image set used for training
    :param image_product: Image product used
    :param embedding_matrix: Embedding matrix for the image set
    :return: Estimated embedding for the input image
    """
    imageProductVector = calculate_image_product_vector(image_input, training_image_set,
                                                        get_image_product(image_product))
    estimateVector = Lagrangian_Method2(embedding_matrix, imageProductVector)[0]
    return estimateVector

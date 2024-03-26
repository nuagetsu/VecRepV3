import numpy as np
from sympy import *
from scipy import optimize


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
    S = np.diag(np.diag(np.dot(A, A.T)))
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

    initial_point_lst = []
    if len(lst_of_eigenvalues) == 1:
        initial_point_lst.append(lst_of_eigenvalues[0] - 0.5)
        initial_point_lst.append(lst_of_eigenvalues[0] + 0.5)
    else:
        initial_point_lst.append(lst_of_eigenvalues[0] - 0.5)
        initial_point_lst.append(lst_of_eigenvalues[-1] + 0.5)
        for i in range(1, len(lst_of_eigenvalues)):
            mid_point = (lst_of_eigenvalues[i] + lst_of_eigenvalues[i - 1]) / 2
            first_quar = (lst_of_eigenvalues[i - 1] + mid_point) / 2
            last_quar = (lst_of_eigenvalues[i] + mid_point) / 2
            initial_point_lst.append(first_quar)
            initial_point_lst.append(mid_point)
            initial_point_lst.append(last_quar)

    # ADD THE MINUS ONE!!!
    equation = H[0] - 1  # use the sympy equation to find the absolute best lambda value

    f = lambdify(lamda, equation, 'numpy')
    z = set()  # solution set (lagrange multipliers)

    # sets different initial points throughout the entire graph
    for initial_point in initial_point_lst:
        solution = optimize.root_scalar(f, x0=initial_point, method="secant")
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

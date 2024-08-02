# Nearest Correlation Matrix
The purpose of the code is to find the nearest correlation matrix of a symmetric matrix.
The correlation matrix finds the correlations between differnet variables, is symmetric and has a unit diagonal (All elements of diagonal is 1).
The nearest correlation matrix is a positive semi-definite matrix (All eigenvalues are more than or equal to 0).

![Example of a Nearest Correlation Matrix](../../assets/PenCorr_demostration.png)

## Intpoint.m

The code first finds a suitable initial point that it can start estimating from in order to use their optimisation algorithm, as not all initial points can be used to locate the optimal solution.

The code uses modified Principle Component Analysis (mPCA) in order to find an ideal starting point.

mPCA is a state-of-the-art method that allows us to isolate the most "important elements" of our data, enabling us to scale up our applications and reduce the dimensions required during analysis.

![Principal component analysis (PCA): Explained and implemented](https://cdn-images-1.medium.com/max/1600/1*V9yJUH9tVrMQI88TuIkCFQ.gif)

## Penalised Method
The penalised method is a method where a new function introduces a matrix G* of rank r, which is the rank that we are trying to reduce our matrix to. 

We construct a new matrix G*, where rank(G*) = r, where r is the rank_constraint we are forcing on the matrix.

This new matrix will be optimised into a nearest correlation matrix using the wide array of known methods currently available, and it is proven that this new matrix has the same optimla solution as our original matrix G.

## Majorisation Method
The majorisation method is a well known method to find the maximum or minimum points located on a curve, by introducing a surrogate function that bounds the curve tightly.

It is an iterative method that will ensure that a local minimum or maximum on a convex curve can be found.

![Majorization Minimization method](https://th.bing.com/th/id/OIP.zpPE6x2LnttA8mPNliDLHgHaCn?w=329&h=123&c=7&r=0&o=5&cb=11&dpr=1.1&pid=1.7)

The surrogate function *(The red curve)* will iteratively go lower and lower until the minimum of the blue curve is reached, which will allow us to find our minimum point.  

# Python Translation

Python translations of several matlab code files in thi folder have been attempted for use in servers or other devices
that cannot run matlab code. Translations of the following files have been attempted:
1. Pencorr.m -> Pencorr.py
2. Intpoint.m -> Inpoint.py
3. CorMat3mex.m -> CorMat3mex.py

These python translations work but cannot produce exactly similar results to the matlab versions due to differences in
implementation of functions that find eigenvectors (since eigenvectors are not unique). The current maximum elementwise
error between results is 1e-6.

Based on testing on large image sets, bugs may still be present within the code. These bugs appear when large image sets
are used with very small rank restrictions (usually less than 20). It is suspected that these bugs are mainly found in the
CorMat3Mex.py file.

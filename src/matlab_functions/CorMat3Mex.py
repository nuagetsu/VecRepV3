import numpy as np
import scipy as sp


def CorMat3Mex(G, b, I, J, OPTIONS, y=None):
    """
    :param G: the given symmetric correlation matrix
    :param b: the right hand side of equality constraints
    :param I: row indices of the fixed elements
    :param J: column indices of the fixed elements
    :param OPTIONS:
    :param y: initial point
    :returns X:  the optimal primal solution
    :returns Y: the optimal dual solution

    Python version of the code in CorMat3Mex.m
    Based on the algorithm in 'A Quadratically Convergent Newton Method for Computing the Nearest Correlation Matrix'
    by Houduo Qi and Defeng Sun
    """
    # set parameters
    tau = 0
    tol = 1.0e-6  # termination tolerance
    tolCG = 1.0e-2  # relative accuracy for CGs
    maxit = 200
    maxitsub = 20  # maximum num of Line Search in Newton method
    maxitCG = 200  # maximum num of iterations in PCG
    sigma = 1.0e-4  # tolerance in the line search of the Newton method
    disp = 1  # display
    const_hist = 5
    progress_test = 1.0e-12

    # get parameters from OPTIONS
    if OPTIONS is not None:
        if 'tau' in OPTIONS:
            tau = OPTIONS['tau']
        if 'tol' in OPTIONS:
            tol = OPTIONS['tol']
        if 'tolCG' in OPTIONS:
            tolCG = OPTIONS['tolCG']
        if 'maxit' in OPTIONS:
            maxit = OPTIONS['maxit']
        if 'maxitsub' in OPTIONS:
            maxitsub = OPTIONS['maxitsub']
        if 'maxitCG' in OPTIONS:
            maxitCG = OPTIONS['maxitCG']
        if 'disp' in OPTIONS:
            disp = OPTIONS['disp']

    n = len(G)
    k = len(b)

    for i in range(k):  # added on November 3, 2009.
        G[I[i], J[i]] = b[i]
        if I[i] != J[i]:
            G[J[i], I[i]] = b[i]

    # reset pars
    G = G - tau * np.eye(n)  # reset G
    G = 0.5 * (G + G.T)  # make G symmetric
    Ind = np.where(I == J)[0]  # reset the diagonal part of b
    b[Ind] = b[Ind] - tau
    b0 = b
    # initial point
    if y is None:
        y = np.zeros(k)
    x0 = y

    if disp:
        print('\n ******************************************************** \n')
        print('          The Semismooth Newton-CG Method                ')
        print('\n ******************************************************** \n')
        print('\n The information of this problem is as follows: \n')
        print(' Dim. of    sdp      constr  = %d \n' % n)
        print(' Num. of equality    constr  = %d \n' % k)

    k1 = 0
    f_eval = 0
    num_pcg = 0
    prec_time = 0
    pcg_time = 0
    eig_time = 0

    f_hist = np.zeros(const_hist)

    X = np.zeros((n, n))
    for i in range(k):
        X[I[i], J[i]] = y[i]
    X = 0.5 * (X + X.T)
    X = G + X
    X = (X + X.T) / 2

    P, lambda_ = MYmexeig(X)
    f0, Fy = gradient(y, I, J, lambda_, P, X, b0)
    f_eval = f_eval + 1

    f = f0
    b = b0 - Fy
    norm_b = np.linalg.norm(b, 2)

    f_hist[0] = f
    val_G = np.sum(np.sum(G * G)) / 2
    # Initial_f = val_G-f0;
    # fprintf('\n Initial Dual Objective Function value  = %d \n', Initial_f)

    while norm_b > tol and k1 < maxit:

        Omega12 = omega_mat(lambda_)

        c = precond_matrix(I, J, Omega12, P)

        d, flag, relres, iterk = pre_cg(b, I, J, tolCG, maxitCG, Omega12, P, c)
        num_pcg = num_pcg + iterk

        slope = np.matmul((Fy - b0).T, d)

        y = x0 + d

        X = np.zeros((n, n))
        for i in range(k):
            X[I[i], J[i]] = y[i]
        X = 0.5 * (X + X.T)
        X = G + X
        X = (X + X.T) / 2

        P, lambda_ = MYmexeig(X)
        f, Fy = gradient(y, I, J, lambda_, P, X, b0)
        f_eval = f_eval + 1

        k_inner = 0
        while k_inner <= maxitsub and f > f0 + sigma * 0.5 ** k_inner * slope + 1.0e-6:

            y = x0 + 0.5 ** k_inner * d  # backtracking

            X = np.zeros((n, n))
            for i in range(k):
                X[I[i], J[i]] = y[i]
            X = 0.5 * (X + X.T)
            X = G + X
            X = (X + X.T) / 2

            P, lambda_ = MYmexeig(X)
            f, Fy = gradient(y, I, J, lambda_, P, X, b0)
            k_inner = k_inner + 1

        k1 = k1 + 1
        f_eval = f_eval + k_inner

        x0 = y
        f0 = f
        b = b0 - Fy
        norm_b = np.linalg.norm(b, 2)

        # slow convergence test
        if k1 < const_hist:
            f_hist[k1] = f
        else:
            for i in range(const_hist - 1):
                f_hist[i] = f_hist[i + 1]
            f_hist[const_hist - 1] = f
        if k1 >= const_hist - 1 and f_hist[0] - f_hist[const_hist - 1] < progress_test:
            print('\n Progress is too slow! :( ')
            break

    # Optimal solution X*
    Ip = np.where(lambda_ > 1.0e-8)[0]
    r = len(Ip)

    if r == 0:
        X = np.zeros((n, n))
    elif r == n:
        X = X
    elif r <= n / 2:
        lambda1 = lambda_[Ip]
        lambda1 = lambda1 ** 0.5
        P1 = P[:, Ip]
        if r > 1:
            P1 = P1 @ np.diag(lambda1)
            X = np.matmul(P1, P1.T)  # Optimal solution X*
        else:
            X = lambda1 ** 2 * np.matmul(P1, P1.T)
    else:
        lambda2 = -lambda_[r:n]  # TODO Check if r or r + 1. For now, r works.
        lambda2 = lambda2 ** 0.5
        P2 = P[:, r:n]
        P2 = np.matmul(P2, np.diag(lambda2))
        X = X + np.matmul(P2, P2.T)  # Optimal solution X*
    X = (X + X.T) / 2

    # optimal primal and dual objective values
    Final_f = val_G - f
    val_obj = np.sum(np.sum((X - G) * (X - G))) / 2
    # convert to original X
    X = X + tau * np.eye(n)

    info = {}
    info['P'] = P
    info['lam'] = np.maximum(0, lambda_)
    info['rank'] = r
    info['numIter'] = k1
    info['numPcg'] = num_pcg
    info['numEig'] = f_eval
    info['eigtime'] = eig_time
    info['pcgtime'] = pcg_time
    info['prectime'] = prec_time
    info['dualVal'] = Final_f

    return X, y, info


# mexeig decomposition
def MYmexeig(X):  # Possible unavoidable error due to non unique eigenvectors
    lambda_, P = np.linalg.eig(X)
    P = np.real(P) * -1
    lambda_ = np.real(lambda_)
    if np.all(np.diff(lambda_) >= 0):
        lambda_ = lambda_[::-1]
        P = P[:, ::-1]
    elif np.all(np.diff(lambda_) >= 0):
        pass
    else:
        lambda_, Inx = np.sort(lambda_)[::-1], np.argsort(lambda_)[::-1]
        P = P[:, Inx]
    return P, lambda_


# To generate F(y)
def gradient(y, I, J, lambda_, P, X, b0):   # Tested wrong!! Check again.
    n = len(P)
    k = len(y)

    const_sparse = 2  # min(5,n/50)

    f = 0.0
    Fy = np.zeros(k)

    I1 = np.where(lambda_ > 1.0e-18)[0]
    r = len(I1)
    if r > 0:
        if r == n:
            f = lambda_.T @ lambda_
            for i in range(k):
                Fy[i] = X[I[i], J[i]]
        elif r <= n / 2:
            lambda1 = lambda_[I1]
            f = np.matmul(lambda1.T, lambda1)

            lambda1 = lambda1 ** 0.5
            P1 = P[:, I1]
            if r > 1:
                P1 = P1 @ np.diag(lambda1)
            else:
                P1 = lambda1[0] * P1        # lambda1 has length 1 since r = 1
            P1T = P1.T

            if k <= const_sparse * n:
                i = 0
                while i < k:
                    Fy[i] = P1[I[i], :] @ P1T[:, J[i]]
                    i = i + 1
            else:
                P = np.matmul(P1, P1T)
                i = 0
                while i < k:  # Same as for loop in other parts
                    Fy[i] = P[I[i], J[i]]
                    i = i + 1
        else:
            lambda2 = -lambda_[r: n]
            f = lambda_.T @ lambda_ - (lambda2.T @ lambda2)
            lambda2 = lambda2 ** 0.5
            P2 = P[:, r: n]
            P2 = P2 @ np.diag(lambda2)
            P2T = P2.T

            if k <= const_sparse * n:
                i = 0
                while i < k:
                    Fy[i] = X[I[i], J[i]] + (P2[I[i], :] @ P2T[:, J[i]])
                    i += 1
            else:
                P = P2 @ P2T
                i = 0
                while i < k:
                    Fy[i] = X[I[i], J[i]] + P[I[i], J[i]]
                    i += 1

    f = 0.5 * f - np.matmul(b0.T, y)
    return f, Fy


# To generate the essential part of the first-order difference of d
def omega_mat(lambda_):     # Tested with 1e-16 error
    n = len(lambda_)
    idx = {'idp': np.where(lambda_ > 0)[0]}
    r = len(idx['idp'])

    if r != 0:
        if r == n:
            Omega12 = np.ones((n, n))
        else:
            s = n - r
            dp = lambda_[:r]
            dn = lambda_[r:n]

            # TODO Check the following function later
            # Omega12 = np.matmul(dp.reshape(-1, 1), np.ones((1, s))) / (np.abs(dp).reshape(-1, 1) + np.abs(dn))
            Omega12 = (dp[:, np.newaxis] * np.ones((1, s))) / (
                        np.abs(dp)[:, np.newaxis] * np.ones((1, s)) + np.ones((r, 1)) * np.abs(dn.T[np.newaxis, :]))

    else:
        Omega12 = np.array([])
    return Omega12


# PCG method
def pre_cg(b, I, J, tol, maxit, Omega12, P, c):         # Tested 1x, uses Jacobian Matrix
    k1 = len(b)
    dim_n, dim_m = P.shape
    flag = 1
    relres = 1000  # give a big value on relres

    r = b  # initial x0=0
    n2b = np.linalg.norm(b, 2)  # norm of b
    tolb = max(tol, min(0.1, n2b)) * n2b  # relative tolerance tol*n2b;   # relative tolerance

    p = np.zeros(k1)

    # preconditioning
    z = r / c  # z = M\r; here M =diag(c); if M is not the identity matrix
    rz1 = np.matmul(r.T, z)
    rz2 = 1
    d = z

    # CG iteration
    for k in range(maxit):
        if k > 1:
            beta = rz1 / rz2
            d = z + beta * d

        w = Jacobian_matrix(d, I, J, Omega12, P)  # W =A(d)

        if k1 > dim_n:  ## if there are more constraints than n
            w = w + 1.0e-2 * min(1.0, 0.1 * n2b) * d  ## perturb it to avoid numerical singularity

        denom = np.matmul(d.T, w)
        relres = np.linalg.norm(r, 2) / n2b  # relative residue=norm(r)/norm(b)

        if denom <= 0:
            p = d / np.linalg.norm(d, 2)  # d is not a descent direction
            break  # exit
        else:
            alpha = rz1 / denom
            p = p + alpha * d
            r = r - alpha * w

        z = r / c  # z = M\r; here M =diag(c); if M is not the identity matrix
        if np.linalg.norm(r, 2) <= tolb:  # Exit if Hp=b solved within the relative tolerance
            relres = np.linalg.norm(r, 2) / n2b  # relative residue =norm(r)/norm(b)
            flag = 0
            break
        rz2 = rz1
        rz1 = np.matmul(r.T, z)

    iterk = k + 1
    return p, flag, relres, iterk


# To generate the Jacobian product with x: F'(y)(x)
def Jacobian_matrix(x, I, J, Omega12, P):
    n = len(P)
    k = len(x)
    r, s = Omega12.shape

    if r == 0:
        Ax = 1.0e-10 * x
    elif r == n:
        Ax = (1 + 1.0e-10) * x
    else:
        Ax = np.zeros(k)
        P1 = P[:, :r]
        P2 = P[:, r:n]

        Z = np.zeros((n, n))
        for i in range(k):
            Z[I[i], J[i]] = x[i]
        Z = 0.5 * (Z + Z.T)

        const_sparse = 2  # min(5,n/50);
        if k <= const_sparse * n:
            # sparse form
            if r < n / 2:
                # H = (Omega.*(P'*sparse(Z)*P))*P';
                H1 = np.matmul(P1.T, Z)  # Can change this to sparse arrays
                Omega12 = np.multiply(Omega12, np.matmul(H1, P2))
                H = np.vstack([np.matmul(H1, P1) @ P1.T + np.matmul(Omega12, P2.T), np.matmul(Omega12.T, P1.T)])

                for i in range(k):
                    Ax[i] = np.matmul(P[I[i], :], H[:, J[i]])
                    Ax[i] = Ax[i] + 1.0e-10 * x[i]  # add a small perturbation
            else:  # if r>=n/2, use a complementary formula.
                # H = ((E-Omega).*(P'*Z*P))*P';
                H2 = np.matmul(P2.T, Z)
                Omega12 = np.ones((r, s)) - Omega12
                Omega12 = np.multiply(Omega12, np.matmul(H2, P1).T)
                H = np.vstack([(np.matmul(Omega12, P2.T)), (np.matmul(Omega12.T, P1.T) + np.matmul(H2, P2) @ P2.T)])

                for i in range(k):  ### AA^* is not the identity matrix
                    if I[i] == J[i]:
                        Ax[i] = x[i] - np.matmul(P[I[i], :], H[:, J[i]])
                    else:
                        Ax[i] = x[i] / 2 - np.matmul(P[I[i], :], H[:, J[i]])
                    Ax[i] = Ax[i] + 1.0e-10 * x[i]

        else:  # dense form
            # Z = full(Z); to use the full form
            # dense form
            if r < n / 2:
                # H = P*(Omega.*(P'*Z*P))*P';
                H1 = np.matmul(P1.T, Z)
                Omega12 = np.multiply(Omega12, np.matmul(H1, P2))
                H = np.matmul(P1, np.matmul(H1, P1.T) + 2.0 * np.matmul(Omega12, P2.T))
                H = (H + H.T) / 2

                for i in range(k):
                    Ax[i] = H[I[i], J[i]]
                    Ax[i] = Ax[i] + 1.0e-10 * x[i]
            else:  # if r>=n/2, use a complementary formula.
                # H = - P*( (E-Omega).*(P'*Z*P) )*P';
                H2 = np.matmul(P2.T, Z)
                Omega12 = np.ones((r, s)) - Omega12
                Omega12 = np.multiply(Omega12, np.matmul(H2, P1).T)
                H = np.matmul(P2, 2.0 * np.matmul(Omega12.T, P1.T) + np.matmul(H2, P2) * P2.T)
                H = (H + H.T) / 2
                H = Z - H

                for i in range(k):  ### AA^* is not the identity matrix
                    Ax[i] = H[I[i], J[i]]
                    Ax[i] = Ax[i] + 1.0e-10 * x[i]

    return Ax


# To generate the (approximate) diagonal preconditioner
def precond_matrix(I, J, Omega12, P):       # Tested 1x
    n = len(P)
    k = len(I)
    r, s = Omega12.shape

    c = np.ones(k)

    H = P.T
    H = H * H
    const_prec = 1
    if r < n:
        if r > 0:
            if k <= const_prec * n:  # compute the exact diagonal preconditioner

                Ind = np.where(I != J)[0]
                k1 = len(Ind)
                if k1 > 0:
                    H1 = np.zeros((n, k1))
                    for i in range(k1):
                        H1[:, i] = np.multiply(P[I[Ind[i]], :].T, P[J[Ind[i]], :].T)

                if r < n / 2:
                    H12 = np.matmul(H[:r, :].T, Omega12)
                    if k1 > 0:
                        H12_1 = np.matmul(H1[:r, :].T, Omega12)

                    d = np.ones(r)

                    # Check matrix multiplication or dot product
                    j = -1
                    for i in range(k):
                        if I[i] == J[i]:
                            c[i] = np.sum(H[:r, I[i]]) * np.matmul(d.T, H[:r, J[i]])
                            c[i] = c[i] + 2.0 * np.matmul(H12[I[i], :], H[r:n, J[i]])
                        else:
                            j += 1
                            c[i] = np.sum(H[:r, I[i]]) * np.matmul(d.T, H[:r, J[i]])
                            c[i] = c[i] + 2.0 * np.matmul(H12[I[i], :], H[r:n, J[i]])
                            c[i] = c[i] + np.sum(H1[:r, j]) * np.matmul(d.T, H1[:r, j])
                            c[i] = c[i] + 2.0 * np.matmul(H12_1[j, :], H1[r:n, j])
                            c[i] = 0.5 * c[i]
                        if c[i] < 1.0e-8:
                            c[i] = 1.0e-8

                else:  # if r>=n/2, use a complementary formula
                    Omega12 = np.ones((r, s)) - Omega12
                    H12 = np.matmul(Omega12, H[r:n, :])
                    if k1 > 0:
                        H12_1 = np.matmul(Omega12, H1[r:n, :])

                    d = np.ones(s)
                    dd = np.ones(n)

                    j = -1
                    for i in range(k):
                        if I[i] == J[i]:
                            c[i] = np.sum(H[r:n, I[i]]) * np.matmul(d.T, H[r:n, J[i]])
                            c[i] = c[i] + 2.0 * np.matmul(H[:r, I[i]].T, H12[:, J[i]])
                            alpha = np.sum(H[:, I[i]])
                            c[i] = alpha * np.matmul(H[:, J[i]].T, dd) - c[i]
                        else:
                            j += 1
                            c[i] = np.sum(H[r:n, I[i]]) * np.matmul(d.T, H[r:n, J[i]])
                            c[i] = c[i] + 2.0 * np.matmul(H[:r, I[i]].T, H12[:, J[i]])
                            alpha = np.sum(H[:, I[i]])
                            c[i] = alpha * np.matmul(H[:, J[i]].T, dd) - c[i]

                            tmp = np.sum(H1[r:n, j]) * np.matmul(d.T, H1[r:n, j])
                            tmp = tmp + 2.0 * np.matmul(H1[:r, j].T, H12_1[:, j])
                            alpha = np.sum(H1[:, j])
                            tmp = alpha * np.matmul(H1[:, j].T, dd) - tmp

                            c[i] = (tmp + c[i]) / 2
                        if c[i] < 1.0e-8:
                            c[i] = 1.0e-8

            else:  # approximate the diagonal preconditioner
                HH1 = H[:r, :]
                HH2 = H[r:n, :]

                if r < n / 2:
                    H0 = np.matmul(HH1.T, Omega12 @ HH2)
                    tmp = np.sum(HH1, axis=0)
                    H0 = H0 + H0.T + np.matmul(tmp.T, tmp)
                else:
                    Omega12 = np.ones(r, s) - Omega12
                    H0 = np.matmul(HH2.T, Omega12.T @ HH1)
                    tmp = np.sum(HH2, axis=0)
                    H0 = H0 + H0.T + np.matmul(tmp.T, tmp)
                    tmp = np.sum(H, axis=0)
                    H0 = np.matmul(tmp.T, tmp) - H0

                for i in range(k):
                    if I[i] == J[i]:
                        c[i] = H0[I[i], J[i]]
                    else:
                        c[i] = 0.5 * H0[I[i], J[i]]
                    if c[i] < 1.0e-8:
                        c[i] = 1.0e-8

    else:  # if r=n
        tmp = np.sum(H, axis=0)
        H0 = np.matmul(np.atleast_2d(tmp).T, np.atleast_2d(tmp))

        for i in range(k):
            if I[i] == J[i]:
                c[i] = H0[I[i], J[i]]
            else:
                c[i] = 0.5 * H0[I[i], J[i]]
            if c[i] < 1.0e-8:
                c[i] = 1.0e-8

    return c

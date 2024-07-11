import numpy as np

def CorMat3Mex(G, b, I, J, OPTIONS, y=None):
    # set parameters
    tau = 0
    tol = 1.0e-6    # termination tolerance
    tolCG = 1.0e-2    # relative accuracy for CGs
    maxit = 200
    maxitsub = 20        # maximum num of Line Search in Newton method
    maxitCG = 200       # maximum num of iterations in PCG
    sigma = 1.0e-4    # tolerance in the line search of the Newton method
    disp = 1         # display
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
    
    t0 = np.array([0, 0, 0, 0, 0, 0])
    
    n = len(G)
    k = len(b)
    
    for i in range(k):    # added on November 3, 2009.
        G[I[i], J[i]] = b[i]
        if I[i] != J[i]:
            G[J[i], I[i]] = b[i]
    
    # reset pars
    G = G - tau * np.eye(n)   # reset G
    G = 0.5 * (G + G.T)         # make G symmetric
    Ind = np.where(I == J)[0]         # reset the diagonal part of b
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
    
    t1 = np.array([0, 0, 0, 0, 0, 0])
    P, lambda_ = MYmexeig(X)
    eig_time = eig_time + (t1 - t1)
    f0, Fy = gradient(y, I, J, lambda_, P, X, b0)
    f_eval = f_eval + 1
    
    f = f0
    b = b0 - Fy
    norm_b = np.linalg.norm(b)
    
    f_hist[0] = f
    val_G = np.sum(np.sum(G * G)) / 2
    # Initial_f = val_G-f0;
    # fprintf('\n Initial Dual Objective Function value  = %d \n', Initial_f)
    
    tt = (t1 - t0)
    hh, mm, ss = time(tt)
    if disp:
        print('\n Iter   NumCGs   StepLen     NormGrad      FunVal  Time')
        print('\n %2.0f        %s         %s         %3.2e    %3.2e    %d:%d:%d' % (0, '-', '-', norm_b, f, hh, mm, ss))
    
    while (norm_b > tol and k1 < maxit):
        
        Omega12 = omega_mat(lambda_)
        
        t2 = np.array([0, 0, 0, 0, 0, 0])
        c = precond_matrix(I, J, Omega12, P)
        prec_time = prec_time + (t2 - t2)
    
        t3 = np.array([0, 0, 0, 0, 0, 0])
        d, flag, relres, iterk = pre_cg(b, I, J, tolCG, maxitCG, Omega12, P, c)
        pcg_time = pcg_time + (t3 - t3)
        num_pcg = num_pcg + iterk
    
        slope = np.dot((Fy - b0).T, d)
    
        y = x0 + d
    
        X = np.zeros((n, n))
        for i in range(k):
            X[I[i], J[i]] = y[i]
        X = 0.5 * (X + X.T)
        X = G + X
        X = (X + X.T) / 2
    
        t1 = np.array([0, 0, 0, 0, 0, 0])
        P, lambda_ = MYmexeig(X)
        eig_time = eig_time + (t1 - t1)
        f, Fy = gradient(y, I, J, lambda_, P, X, b0)
        f_eval = f_eval + 1
    
        k_inner = 0
        while (k_inner <= maxitsub and f > f0 + sigma * 0.5 ** k_inner * slope + 1.0e-6):
    
            y = x0 + 0.5 ** k_inner * d  # backtracking
    
            X = np.zeros((n, n))
            for i in range(k):
                X[I[i], J[i]] = y[i]
            X = 0.5 * (X + X.T)
            X = G + X
            X = (X + X.T) / 2
    
            t1 = np.array([0, 0, 0, 0, 0, 0])
            P, lambda_ = MYmexeig(X)
            eig_time = eig_time + (t1 - t1)
            f, Fy = gradient(y, I, J, lambda_, P, X, b0)
            k_inner = k_inner + 1
    
        k1 = k1 + 1
        f_eval = f_eval + k_inner
    
        x0 = y
        f0 = f
        b = b0 - Fy
        norm_b = np.linalg.norm(b)
    
        tt = (t1 - t0)
        hh, mm, ss = time(tt)
        if disp:
            print('\n       %2.0d     %2.0d      %3.2e    %3.2e    %3.2e    %d:%d:%d' % (k1, iterk, 0.5 ** k_inner, norm_b, f, hh, mm, ss))
    
        # slow convergence test
        if (k1 < const_hist):
            f_hist[k1 + 1] = f
        else:
            for i in range(const_hist - 1):
                f_hist[i] = f_hist[i + 1]
            f_hist[const_hist - 1] = f
        if (k1 >= const_hist - 1 and f_hist[0] - f_hist[const_hist - 1] < progress_test):
            print('\n Progress is too slow! :( ')
            break
    
    # Optimal solution X*
    Ip = np.where(lambda_ > 1.0e-8)[0]
    r = len(Ip)
    
    if (r == 0):
        X = np.zeros((n, n))
    elif (r == n):
        X = X
    elif (r <= n / 2):
        lambda1 = lambda_[Ip]
        lambda1 = lambda1 ** 0.5
        P1 = P[:, Ip]
        if r > 1:
            P1 = P1 * np.diag(lambda1)
            X = np.dot(P1, P1.T)  # Optimal solution X*
        else:
            X = lambda1 ** 2 * np.dot(P1, P1.T)
    else:
        lambda2 = -lambda_[r + 1:n]
        lambda2 = lambda2 ** 0.5
        P2 = P[:, r + 1:n]
        P2 = P2 * np.diag(lambda2)
        X = X + np.dot(P2, P2.T)  # Optimal solution X*
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
    
    time_used = np.sum(tt)
    if disp:
        print('\n')
        print('\n ================ Final Information ================= \n')
        print(' Total number of iterations      = %2.0f \n' % k1)
        print(' Number of func. evals(mexeig)   = %2.0f \n' % f_eval)
        print(' Number of CG Iterations         = %2.0f \n' % num_pcg)
        print(' Primal objective value          = %d \n' % val_obj)
        print(' Dual objective value            = %d \n' % Final_f)
        print(' Norm of Gradient                = %3.2e \n' % norm_b)
        print(' Rank of X-tau*I                 ====== %8.0f \n' % r)
        print(' Computing time for preconditioner     = %3.1f \n' % prec_time)
        print(' Computing time for CG iterations      = %3.1f \n' % pcg_time)
        print(' Computing time for eigen-decom        = %3.1f \n' % eig_time)
        print(' Total Computing time (secs)           = %3.1f \n' % time_used)
        print(' ====================================================== \n')
    
    return X, y, info

# To change the format of time
def time(t):
    t = round(t)
    h = int(t / 3600)
    m = int((t % 3600) / 60)
    s = int((t % 60) % 60)
    return h, m, s

# mexeig decomposition
def MYmexeig(X):
    P, lambda_ = np.linalg.eig(X)
    P = np.real(P)
    lambda_ = np.real(lambda_)
    if np.all(np.sort(lambda_) == lambda_):
        lambda_ = lambda_[::-1]
        P = P[:, ::-1]
    else:
        lambda_, Inx = np.sort(lambda_)[::-1], np.argsort(lambda_)[::-1]
        P = P[:, Inx]
    return P, lambda_

# To generate F(y)
def gradient(y, I, J, lambda_, P, X, b0):
    n = len(P)
    k = len(y)
    
    const_sparse = 2  # min(5,n/50)
    
    f = 0.0
    Fy = np.zeros(k)
    
    I1 = np.where(lambda_ > 1.0e-18)[0]
    r = len(I1)
    if r > 0:
        if r == n:
            f = np.dot(lambda_, lambda_)
            for i in range(k):
                Fy[i] = X[I[i], J[i]]
        elif r <= n / 2:
            lambda1 = lambda_[I1]
            f = np.dot(lambda1, lambda1)
    
            lambda1 = lambda1 ** 0.5
            P1 = P[:, I1]
            if r > 1:
                P1 = np.dot(P1, np.diag(lambda1))
                for i in range(k):
                    Fy[i] = np.dot(P1[I[i], :], P1[:, J[i]])
            else:
                Fy = lambda1 ** 2 * np.dot(P1[I, :], P1[:, J])
        else:
            lambda2 = -lambda_[r + 1:n]
            f = np.dot(lambda_, lambda_) - np.dot(lambda2, lambda2)
    
            lambda2 = lambda2 ** 0.5
            P2 = P[:, r + 1:n]
            P2 = np.dot(P2, np.diag(lambda2))
            Fy = X[I, J] + np.dot(P2[I, :], P2[:, J])
    
    f = 0.5 * f - np.dot(b0.T, y)
    return f, Fy

# To generate the essential part of the first-order difference of d
def omega_mat(lambda_):
    n = len(lambda_)
    idx = {}
    idx['idp'] = np.where(lambda_ > 0)[0]
    r = len(idx['idp'])
    
    if r != 0:
        if r == n:
            Omega12 = np.ones((n, n))
        else:
            s = n - r
            dp = lambda_[:r]
            dn = lambda_[r:n]
    
            Omega12 = np.dot(dp.reshape(-1, 1), np.ones((1, s))) / (np.abs(dp).reshape(-1, 1) + np.abs(dn))
    else:
        Omega12 = np.array([])
    return Omega12

# PCG method
def pre_cg(b, I, J, tol, maxit, Omega12, P, c):
    k1 = len(b)
    dim_n, dim_m = P.shape
    flag = 1
    relres = 1000  # give a big value on relres
    
    r = b  # initial x0=0
    n2b = np.linalg.norm(b)  # norm of b
    tolb = max(tol, min(0.1, n2b)) * n2b  # relative tolerance tol*n2b;   # relative tolerance
    
    p = np.zeros(k1)
    
    # preconditioning
    z = r / c  # z = M\r; here M =diag(c); if M is not the identity matrix
    rz1 = np.dot(r.T, z)
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
    
        denom = np.dot(d.T, w)
        relres = np.linalg.norm(r) / n2b  # relative residue=norm(r)/norm(b)
    
        if denom <= 0:
            p = d / np.linalg.norm(d)  # d is not a descent direction
            break  # exit
        else:
            alpha = rz1 / denom
            p = p + alpha * d
            r = r - alpha * w
    
        z = r / c  # z = M\r; here M =diag(c); if M is not the identity matrix
        if np.linalg.norm(r) <= tolb:  # Exit if Hp=b solved within the relative tolerance
            relres = np.linalg.norm(r) / n2b  # relative residue =norm(r)/norm(b)
            flag = 0
            break
        rz2 = rz1
        rz1 = np.dot(r.T, z)
    
    iterk = k
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
                H1 = np.dot(P1.T, Z)
                Omega12 = Omega12 * np.dot(H1, P2)
                H = np.vstack([(np.dot(H1, P1) * P1.T + np.dot(Omega12, P2.T)).T, (np.dot(Omega12.T, P1)).T])
    
                for i in range(k):
                    Ax[i] = np.dot(P[I[i], :], H[:, J[i]])
                    Ax[i] = Ax[i] + 1.0e-10 * x[i]  # add a small perturbation
            else:  # if r>=n/2, use a complementary formula.
                # H = ((E-Omega).*(P'*Z*P))*P';
                H2 = np.dot(P2.T, Z)
                Omega12 = 1 - Omega12
                Omega12 = Omega12 * np.dot(H2, P1.T)
                H = np.vstack([(np.dot(Omega12, P2.T)).T, (np.dot(Omega12.T, P1.T) + np.dot(H2, P2) * P2.T).T])
    
                for i in range(k):  ### AA^* is not the identity matrix
                    if I[i] == J[i]:
                        Ax[i] = x[i] - np.dot(P[I[i], :], H[:, J[i]])
                    else:
                        Ax[i] = x[i] / 2 - np.dot(P[I[i], :], H[:, J[i]])
                    Ax[i] = Ax[i] + 1.0e-10 * x[i]
    
        else:  # dense form
            # Z = full(Z); to use the full form
            # dense form
            if r < n / 2:
                # H = P*(Omega.*(P'*Z*P))*P';
                H1 = np.dot(P1.T, Z)
                Omega12 = Omega12 * np.dot(H1, P2)
                H = np.dot(P1, np.dot(H1, P1.T) + 2.0 * np.dot(Omega12, P2.T))
                H = (H + H.T) / 2
    
                for i in range(k):
                    Ax[i] = H[I[i], J[i]]
                    Ax[i] = Ax[i] + 1.0e-10 * x[i]
            else:  # if r>=n/2, use a complementary formula.
                # H = - P*( (E-Omega).*(P'*Z*P) )*P';
                H2 = np.dot(P2.T, Z)
                Omega12 = 1 - Omega12
                H = np.dot(P2, 2.0 * np.dot(Omega12.T, P1.T) + np.dot(H2, P2) * P2.T)
                H = (H + H.T) / 2
                H = Z - H
    
                for i in range(k):  ### AA^* is not the identity matrix
                    Ax[i] = H[I[i], J[i]]
                    Ax[i] = Ax[i] + 1.0e-10 * x[i]
    
    return Ax

# To generate the (approximate) diagonal preconditioner
def precond_matrix(I, J, Omega12, P):
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
                        H1[:, i] = P[I[Ind[i]], :].T * P[J[Ind[i]], :]
    
                if r < n / 2:
                    H12 = np.dot(H[:r, :].T, Omega12)
                    if k1 > 0:
                        H12_1 = np.dot(H1[:r, :].T, Omega12)
    
                    d = np.ones(r)
    
                    for i in range(k):
                        if I[i] == J[i]:
                            c[i] = np.sum(H[:r, I[i]]) * np.dot(d.T, H[:r, J[i]])
                            c[i] = c[i] + 2.0 * np.dot(H12[I[i], :], H[r:n, J[i]])
                        else:
                            c[i] = np.sum(H[:r, I[i]]) * np.dot(d.T, H[:r, J[i]])
                            c[i] = c[i] + 2.0 * np.dot(H12[I[i], :], H[r:n, J[i]])
                            c[i] = c[i] + np.sum(H1[:r, i]) * np.dot(d.T, H1[:r, i])
                            c[i] = c[i] + 2.0 * np.dot(H12_1[i, :], H1[r:n, i])
                            c[i] = 0.5 * c[i]
                        if c[i] < 1.0e-8:
                            c[i] = 1.0e-8
    
                else:  # if r>=n/2, use a complementary formula
                    Omega12 = 1 - Omega12
                    H12 = np.dot(Omega12, H[r:n, :])
                    if k1 > 0:
                        H12_1 = np.dot(Omega12, H1[r:n, :])
    
                    d = np.ones(s)
                    dd = np.ones(n)
    
                    for i in range(k):
                        if I[i] == J[i]:
                            c[i] = np.sum(H[r:n, I[i]]) * np.dot(d.T, H[r:n, J[i]])
                            c[i] = c[i] + 2.0 * np.dot(H[:r, I[i]].T, H12[:, J[i]])
                            alpha = np.sum(H[:, I[i]])
                            c[i] = alpha * np.dot(H[:, J[i]].T, dd) - c[i]
                        else:
                            c[i] = np.sum(H[r:n, I[i]]) * np.dot(d.T, H[r:n, J[i]])
                            c[i] = c[i] + 2.0 * np.dot(H[:r, I[i]].T, H12[:, J[i]])
                            alpha = np.sum(H[:, I[i]])
                            c[i] = alpha * np.dot(H[:, J[i]].T, dd) - c[i]
    
                            tmp = np.sum(H1[r:n, i]) * np.dot(d.T, H1[r:n, i])
                            tmp = tmp + 2.0 * np.dot(H1[:r, i].T, H12_1[:, i])
                            alpha = np.sum(H1[:, i])
                            tmp = alpha * np.dot(H1[:, i].T, dd) - tmp
    
                            c[i] = (tmp + c[i]) / 2
                        if c[i] < 1.0e-8:
                            c[i] = 1.0e-8
    
            else:  # approximate the diagonal preconditioner
                HH1 = H[:r, :]
                HH2 = H[r:n, :]
    
                if r < n / 2:
                    H0 = np.dot(HH1.T, Omega12 * HH2)
                    tmp = np.sum(HH1, axis=0)
                    H0 = H0 + H0.T + np.dot(tmp.T, tmp)
                else:
                    Omega12 = 1 - Omega12
                    H0 = np.dot(HH2.T, (Omega12).T * HH1)
                    tmp = np.sum(HH2, axis=0)
                    H0 = H0 + H0.T + np.dot(tmp.T, tmp)
                    tmp = np.sum(H, axis=0)
                    H0 = np.dot(tmp.T, tmp) - H0
    
            for i in range(k):
                if I[i] == J[i]:
                    c[i] = H0[I[i], J[i]]
                else:
                    c[i] = 0.5 * H0[I[i], J[i]]
                if c[i] < 1.0e-8:
                    c[i] = 1.0e-8
    
    else:  # if r=n
        tmp = np.sum(H, axis=0)
        H0 = np.dot(tmp.T, tmp)
    
        for i in range(k):
            if I[i] == J[i]:
                c[i] = H0[I[i], J[i]]
            else:
                c[i] = 0.5 * H0[I[i], J[i]]
            if c[i] < 1.0e-8:
                c[i] = 1.0e-8
    
    return c

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
                H1 = np.dot(P1.T, Z)
                Omega12 = Omega12 * np.dot(H1, P2)
                H = np.vstack([(np.dot(H1, P1) * P1.T + np.dot(Omega12, P2.T)).T, (np.dot(Omega12.T, P1)).T])
    
                for i in range(k):
                    Ax[i] = np.dot(P[I[i], :], H[:, J[i]])
                    Ax[i] = Ax[i] + 1.0e-10 * x[i]  # add a small perturbation
            else:  # if r>=n/2, use a complementary formula.
                # H = ((E-Omega).*(P'*Z*P))*P';
                H2 = np.dot(P2.T, Z)
                Omega12 = 1 - Omega12
                H = np.vstack([(np.dot(Omega12, P2.T)).T, (np.dot(Omega12.T, P1.T) + np.dot(H2, P2) * P2.T).T])
    
                for i in range(k):  ### AA^* is not the identity matrix
                    if I[i] == J[i]:
                        Ax[i] = x[i] - np.dot(P[I[i], :], H[:, J[i]])
                    else:
                        Ax[i] = x[i] / 2 - np.dot(P[I[i], :], H[:, J[i]])
                    Ax[i] = Ax[i] + 1.0e-10 * x[i]
    
        else:  # dense form
            # Z = full(Z); to use the full form
            # dense form
            if r < n / 2:
                # H = P*(Omega.*(P'*Z*P))*P';
                H1 = np.dot(P1.T, Z)
                Omega12 = Omega12 * np.dot(H1, P2)
                H = np.dot(P1, np.dot(H1, P1.T) + 2.0 * np.dot(Omega12, P2.T))
                H = (H + H.T) / 2
    
                for i in range(k):
                    Ax[i] = H[I[i], J[i]]
                    Ax[i] = Ax[i] + 1.0e-10 * x[i]
            else:  # if r>=n/2, use a complementary formula.
                # H = - P*( (E-Omega).*(P'*Z*P) )*P';
                H2 = np.dot(P2.T, Z)
                Omega12 = 1 - Omega12
                H = np.dot(P2, 2.0 * np.dot(Omega12.T, P1.T) + np.dot(H2, P2) * P2.T)
                H = (H + H.T) / 2
                H = Z - H
    
                for i in range(k):  ### AA^* is not the identity matrix
                    Ax[i] = H[I[i], J[i]]
                    Ax[i] = Ax[i] + 1.0e-10 * x[i]
    
    return Ax

# To generate the (approximate) diagonal preconditioner
def precond_matrix(I, J, Omega12, P):
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
                        H1[:, i] = P[I[Ind[i]], :].T * P[J[Ind[i]], :]
    
                if r < n / 2:
                    H12 = H[:r, :].T * Omega12
                    if k1 > 0:
                        H12_1 = H1[:r, :].T * Omega12
    
                    d = np.ones(r)
    
                    for i in range(k):
                        if I[i] == J[i]:
                            c[i] = np.sum(H[:r, I[i]]) * np.dot(d.T, H[:r, J[i]])
                            c[i] = c[i] + 2.0 * np.dot(H12[I[i], :], H[r:n, J[i]])
                        else:
                            c[i] = np.sum(H[:r, I[i]]) * np.dot(d.T, H[:r, J[i]])
                            c[i] = c[i] + 2.0 * np.dot(H12[I[i], :], H[r:n, J[i]])
                            c[i] = c[i] + np.sum(H1[:r, i]) * np.dot(d.T, H1[:r, i])
                            c[i] = c[i] + 2.0 * np.dot(H12_1[i, :], H1[r:n, i])
                            c[i] = 0.5 * c[i]
                        if c[i] < 1.0e-8:
                            c[i] = 1.0e-8
    
                else:  # if r>=n/2, use a complementary formula
                    Omega12 = 1 - Omega12
                    H12 = Omega12 * H[r:n, :]
                    if k1 > 0:
                        H12_1 = Omega12 * H1[r:n, :]
    
                    d = np.ones(s)
                    dd = np.ones(n)
    
                    for i in range(k):
                        if I[i] == J[i]:
                            c[i] = np.sum(H[r:n, I[i]]) * np.dot(d.T, H[r:n, J[i]])
                            c[i] = c[i] + 2.0 * np.dot(H[:r, I[i]].T, H12[:, J[i]])
                            alpha = np.sum(H[:, I[i]])
                            c[i] = alpha * np.dot(H[:, J[i]].T, dd) - c[i]
                        else:
                            c[i] = np.sum(H[r:n, I[i]]) * np.dot(d.T, H[r:n, J[i]])
                            c[i] = c[i] + 2.0 * np.dot(H[:r, I[i]].T, H12[:, J[i]])
                            alpha = np.sum(H[:, I[i]])
                            c[i] = alpha * np.dot(H[:, J[i]].T, dd) - c[i]
    
                            tmp = np.sum(H1[r:n, i]) * np.dot(d.T, H1[r:n, i])
                            tmp = tmp + 2.0 * np.dot(H1[:r, i].T, H12_1[:, i])
                            alpha = np.sum(H1[:, i])
                            tmp = alpha * np.dot(H1[:, i].T, dd) - tmp
    
                            c[i] = (tmp + c[i]) / 2
                        if c[i] < 1.0e-8:
                            c[i] = 1.0e-8
    
            else:  # approximate the diagonal preconditioner
                HH1 = H[:r, :]
                HH2 = H[r:n, :]
    
                if r < n / 2:
                    H0 = HH1.T * Omega12 * HH2
                    tmp = np.sum(HH1, axis=0)
                    H0 = H0 + H0.T + tmp.T * tmp
                else:
                    Omega12 = 1 - Omega12
                    H0 = HH2.T * (Omega12.T * HH1)
                    tmp = np.sum(HH2, axis=0)
                    H0 = H0 + H0.T + tmp.T * tmp
                    tmp = np.sum(H, axis=0)
                    H0 = tmp.T * tmp - H0
    
            for i in range(k):
                if I[i] == J[i]:
                    c[i] = H0[I[i], J[i]]
                else:
                    c[i] = 0.5 * H0[I[i], J[i]]
                if c[i] < 1.0e-8:
                    c[i] = 1.0e-8
    
    else:  # if r=n
        tmp = np.sum(H, axis=0)
        H0 = tmp.T * tmp
    
        for i in range(k):
            if I[i] == J[i]:
                c[i] = H0[I[i], J[i]]
            else:
                c[i] = 0.5 * H0[I[i], J[i]]
            if c[i] < 1.0e-8:
                c[i] = 1.0e-8
    
    return c



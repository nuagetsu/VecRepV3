import numpy as np
import scipy as sp

from src.matlab_functions.CorMat3Mex import CorMat3Mex


def IntPoint(G, e, I_e, J_e, Rank, X, P, lmbda, opt_disp):
    """
    Finds the initial point to be used in pencorr algorithm.
    """
    G = (G + G.T)/2
    k_e = len(e)
    Ind = np.where(I_e == J_e)[0]
    e_diag = e[Ind]
    
    tolinf = 1.0e-6
    rankErr_stop = 1.0e-1
    maxit = 5
    maxitsub = 2
    const_disp1 = 20
    const_disp2 = 10
    infoNum = {"callCN": 0, "iterCN": 0, "CG": 0, "eigendecom": 0}
    rho = 1.0e2
    rho_step = 10
    rho_max = 1.0e8

    # Generate initial Z
    rank_X = np.count_nonzero(lmbda > 1.0e-8)
    rankErr = abs(np.sum(lmbda) - np.sum(lmbda[:Rank]))
    rankErr0 = rankErr
    Z = mPCA(P, lmbda, Rank, e_diag)

    # Test how good is mPCA(X)
    infeas = np.zeros(k_e)
    for i in range(k_e):
        infeas[i] = e[i] - Z[I_e[i], J_e[i]]
    normInf = np.linalg.norm(infeas, 2)
    
    if normInf <= tolinf:
        X = Z
        P, lmbda = MYmexeig(X, 0)
        rank_X = Rank
        rankErr = abs(np.sum(lmbda) - np.sum(lmbda[:Rank]))
        residue_X = np.sqrt(np.sum((X - G)*(X - G)))
        infoNum["eigendecom"] += 1
        return X, P, lmbda, rank_X, rankErr, normInf, infoNum

    # Initial Residue
    residue_X = np.sqrt(np.sum((X - G)*(X - G)))
    residue_Z = np.sqrt(np.sum((Z - G)*(Z - G)))
    residue_XZ = np.sqrt(np.sum((X - Z)*(X - Z)))
    fc = 0.5*residue_X**2 + 0.5*residue_Z**2
    fc += 0.5*rho*residue_XZ**2
    
    opts = {"disp": 0}
    break_level = 0
    total_AltProj = 0
    
    for k1 in range(maxit):
        
        for itersub in range(maxitsub):
            C = (X - G)
            G0 = G + rho*Z - C
            G0 = G0/(1 + rho)
            z = np.zeros(k_e)
            for i in range(k_e):
                z[i] = e[i] - G0[I_e[i], J_e[i]]

            X, z, info = CorMat3Mex(G0, e, I_e, J_e, opts, y=z)
            P = info.P
            lmbda = info.lam
            rank_X = info.rank
            rankErr = abs(np.sum(lmbda) - np.sum(lmbda[:Rank]))
            infoNum["callCN"] += 1
            infoNum["iterCN"] += info.numIter
            infoNum["CG"] += info.numPcg
            infoNum["eigendecom"] += info.numEig
            
            residue_X = np.sqrt(np.sum((X - G)*(X - G)))
            residue_XZ = np.sqrt(np.sum((X - Z)*(X - Z)))
            fc = 0.5*residue_X**2 + 0.5*residue_Z**2
            fc += rho*0.5*residue_XZ**2
            
            if rankErr <= rankErr_stop * rankErr0:
                break_level = 1
                break

            # Update Z: project onto the rank constraint
            C = (Z - G)
            G0 = G + rho*X - C
            G0 = G0/(1 + rho)
            P, lmbda = MYmexeig(G0, 1)
            Z = Projr(P, lmbda, Rank)
            infoNum["eigendecom"] += 1
            
            infeas = np.zeros(k_e)
            for i in range(k_e):
                infeas[i] = e[i] - Z[I_e[i], J_e[i]]
            normInf = np.linalg.norm(infeas, 2)

            
            if normInf <= tolinf:
                X = Z
                P, lmbda = MYmexeig(X, 0)
                infoNum["eigendecom"] += 1
                rank_X = Rank
                rankErr = abs(np.sum(lmbda) - np.sum(lmbda[:Rank]))
                residue_X = np.sqrt(np.sum((X - G)*(X - G)))
                break_level = 2
                break
        
        total_AltProj += itersub
        
        if break_level == 1:
            print('\n Alternating terminates at projection onto the linear constraints with small rankErr!')
            break
        elif break_level == 2:
            print('\n Alternating terminates at projection onto rank constraints with small normInf!')
            print('\n Norm of infeasibility = %4.3e' % normInf)
            break
        else:
            rho = min(rho_step*rho, rho_max)

    return X, P, lmbda, rank_X, rankErr, normInf, infoNum


def MYmexeig(X, order_abs):
    lambda_, P = np.linalg.eig(X)
    P = np.real(P)
    lambda_ = np.real(lambda_)
    if order_abs == 0:
        if np.all(np.diff(lambda_) >= 0):
            lambda_ = lambda_[::-1]     # TODO Check
            P = P[:, ::-1]
        elif np.all(np.sort(lambda_)[::-1]):
            return P, lambda_
        else:
            Inx = np.argsort(lambda_)[::-1]
            lambda_ = lambda_[Inx]
            P = P[:, Inx]
    elif order_abs == 1:
        if np.all(np.sort(np.abs(lambda_))):
            lambda_ = lambda_[::-1]
            P = P[:, ::-1]
        elif np.all(np.sort(np.abs(lambda_[::-1]))):
            return P, lambda_
        else:
            Inx = np.argsort(np.abs(lambda_))[::-1]
            lambda_ = lambda_[Inx]
            P = P[:, Inx]
    return P, lambda_


def mPCA(P, lmbda, Rank, b):
    n = len(lmbda)
    if b is None:     # Unsure on nargin
        b = np.ones(n)
    
    if Rank > 0:
        P1 = P[:, :Rank]
        lambda1 = lmbda[:Rank]
        lambda1 = np.sqrt(lambda1)
        if Rank > 1:
            P1 = P1 @ np.diag(lambda1)
        else:
            P1 = P1 @ lambda1
        pert_Mat = np.random.rand(n, Rank)
        for i in range(n):
            s = np.linalg.norm(P1[i, :], 2)
            if s < 1.0e-12:
                P1[i, :] = pert_Mat[i, :]
                s = np.linalg.norm(P1[i, :], 2)
            P1[i, :] = P1[i, :] / s
            P1[i, :] = P1[i, :] * np.sqrt(b[i])
        X = P1 @ P1.T
    else:
        X = np.zeros((n, n))
    return X


def Projr(P, lambda_, r):
    n = len(lambda_)
    X = np.zeros((n, n))
    if r > 0:
        P1 = P[:, :r]
        lambda1 = lambda_[:r]
        for i in range(r):
            P1[:, i] = lambda1[i] @ P1[:, i]
        X = P[:, :r] @ P1.T
    return X


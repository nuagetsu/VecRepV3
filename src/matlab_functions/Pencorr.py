import numpy as np
from numpy.linalg import eigh
import scipy as sp

from src.matlab_functions.CorMat3Mex import CorMat3Mex
from src.matlab_functions.IntPoint import IntPoint


def PenCorr(G, ConstrA, Rank, OPTIONS):
    
    
    # get constraints infos from constrA
    e = ConstrA['e']
    I_e = ConstrA['Ie']
    J_e = ConstrA['Je']
    k_e = len(e)
    n = len(G)
    
    tolrel = 1.0e-5
    
    # get parameters from the OPTIONS structure
    if 'tau' in OPTIONS:
        tau = OPTIONS['tau']
    else:
        tau = 0
    if 'tolrel' in OPTIONS:
        tolrel = OPTIONS['tolrel']
    if 'tolrank' in OPTIONS:
        tolrank = OPTIONS['tolrank']
    if 'tolsub' in OPTIONS:
        tolsub = OPTIONS['tolsub']
    if 'tolPCA' in OPTIONS:
        tolPCA = OPTIONS['tolPCA']
    if 'tolinfeas' in OPTIONS:
        tolinfeas = OPTIONS['tolinfeas']
    if 'tolsub_rank' in OPTIONS:
        tolsub_rank = OPTIONS['tolsub_rank']
    if 'maxit' in OPTIONS:
        maxit = OPTIONS['maxit']
    if 'maxitsub' in OPTIONS:
        maxitsub = OPTIONS['maxitsub']
    if 'use_CorNewtonPCA' in OPTIONS:
        use_CorNewtonPCA = OPTIONS['use_CorNewtonPCA']
    if 'use_InitialPoint' in OPTIONS:
        use_InitialPoint = OPTIONS['use_InitialPoint']
    
    tau = 0
    innerTolrel = tolrel
    tolsub = max(innerTolrel, tolrel)
    tolPCA = max(innerTolrel, tolrel)
    tolinfeas = max(innerTolrel, tolrel)
    tolsub_rank = tolsub
    tolrank = 1.0e-8
    maxit = 500
    maxitsub = 100
    use_InitialPoint = 1
    finalPCA = 0
    residue_cutoff = 10
    if tolrel <= 1.0e-4:
        residue_cutoff = 100

    
    # reset input pars
    G = G - tau * sp.eye(n)
    G = (G + G.T) / 2
    Ind = np.where(I_e == J_e)[0]
    e[Ind] = e[Ind] - tau
    e_diag = e[Ind]
    
    # constant pars
    const_disp1 = 10
    const_disp2 = 10
    const_rank_hist = 5
    const_rank_hist = max(2, const_rank_hist)
    const_rankErr_hist = 3
    const_rankErr_hist = max(2, const_rankErr_hist)
    const_funcVal_hist = 2
    const_funcVal_hist = max(2, const_funcVal_hist)
    const_residue_hist = const_funcVal_hist
    const_residue_hist = max(2, const_residue_hist)
    rank_hist = np.zeros(const_rank_hist)
    rankErr_hist = np.zeros(const_rankErr_hist)
    funcVal_hist = np.zeros(const_funcVal_hist)
    residue_hist = np.zeros(const_residue_hist)
    progress_rank1 = 1
    progress_rank2 = 1
    progress_relErr = 1.0e-5
    progress_rankErr = 1.0e-3
    
    # penalty pars
    c0_min = 1.0
    c0_max = 1e2
    alpha_min = 1.2
    alpha_max = 4.0
    c_max = 1.0e8
    
    Totalcall_CN = 0
    Totaliter_CN = 0
    Totalnumb_CG = 0
    Totalnumb_eigendecom = 0
    
    # CorNewton3Mex preprocessing
    y = np.zeros(k_e)
    for i in range(k_e):
        y[i] = e[i] - G[I_e[i], J_e[i]]
    opts = {'disp': 0}
    X, y, info = CorMat3Mex(G, e, I_e, J_e, opts, y)
    P = info['P']
    lambda_ = info['lam']
    rank_X = info['rank']
    Totalcall_CN = Totalcall_CN + 1
    Totaliter_CN = Totaliter_CN + info['numIter']
    Totalnumb_CG = Totalnumb_CG + info['numPcg']
    Totalnumb_eigendecom = Totalnumb_eigendecom + info['numEig']
    residue_CorNewton = np.sum(np.sum((X - G) * (X - G)))
    residue_CorNewton = np.sqrt(residue_CorNewton)
    rankErr_CorNewton = np.abs(np.sum(lambda_) - np.sum(lambda_[:Rank]))
    if rankErr_CorNewton <= tolrank:
        if Rank < n:
            INFOS = {'iter': 0, 'callCN': Totalcall_CN, 'itCN': Totaliter_CN, 'itCG': Totalnumb_CG, 'numEig': Totalnumb_eigendecom, 'rank': rank_X, 'rankErr': rankErr_CorNewton, 'residue': residue_CorNewton}
            return X, INFOS
    
    residue_1 = residue_CorNewton
    
    # check how good is CorNewton_PCA
    X1 = mPCA(P, lambda_, Rank, e_diag)
    residue_CorNewtonPCA = np.sum(np.sum((X1 - G) * (X1 - G)))
    residue_CorNewtonPCA = np.sqrt(residue_CorNewtonPCA)
    residue_error = np.abs(residue_CorNewtonPCA - residue_CorNewton)
    infeas = np.zeros(k_e)
    for i in range(k_e):
        infeas[i] = e[i] - X1[I_e[i], J_e[i]]
    NormInf_CorNewtonPCA = np.linalg.norm(infeas)
    if residue_error / max(residue_cutoff, residue_CorNewtonPCA) <= tolPCA and NormInf_CorNewtonPCA <= tolinfeas:
        INFOS = {'iter': 0, 'callCN': Totalcall_CN, 'itCN': Totaliter_CN, 'itCG': Totalnumb_CG, 'numEig': Totalnumb_eigendecom, 'rank': rank_X, 'rankErr': 0, 'residue': residue_CorNewtonPCA}
        return X1, INFOS
    
    if use_InitialPoint:
        opt_disp = 1
        X, P, lambda_, rank_X, rankErr, normInf, infoNum = IntPoint(G, e, I_e, J_e, Rank, X, P, lambda_, opt_disp)
        Totalcall_CN = Totalcall_CN + infoNum['callCN']
        Totaliter_CN = Totaliter_CN + infoNum['iterCN']
        Totalnumb_CG = Totalnumb_CG + infoNum['CG']
        Totalnumb_eigendecom = Totalnumb_eigendecom + infoNum['eigendecom']
        residue_int = np.sum(np.sum((X - G) * (X - G)))
        residue_int = np.sqrt(residue_int)
        residue_1 = residue_int
    else:
        X = X1
        P, lambda_ = MYmexeig(X)
        Totalnumb_eigendecom = Totalnumb_eigendecom + 1
        residue_int = residue_CorNewtonPCA
        residue_1 = residue_int
    
    # initialize U
    P1 = P[:, :Rank]
    U = np.dot(P1, P1.T)
    rankErr = np.abs(np.sum(lambda_) - np.sum(lambda_[:Rank]))
    
    # initial penalty parameter c
    if use_InitialPoint:
        c0 = 0.50 * (residue_int ** 2 - residue_CorNewton ** 2)
        c0 = 0.25 * c0 / max(1.0, rankErr_CorNewton - rankErr)
    else:
        c0 = 0.50 * (residue_CorNewtonPCA ** 2 - residue_CorNewton ** 2)
        c0 = 0.25 * c0 / max(1.0, rankErr_CorNewton)
    if tolrel >= 1.0e-1:
        c0 = 4 * c0
    elif tolrel >= 1.0e-2:
        c0 = 2 * c0
    c0 = max(c0, c0_min)
    c0 = min(c0, c0_max)
    c = c0
    
    relErr_0 = 1.0e6
    break_level = 0
    
    k1 = 1
    sum_iter = 0
    while k1 <= maxit:
        subtotaliter_CN = 0
        subtotalnumb_CG = 0
        subtotalnumb_eigendecom = 0
        
        fc = 0.5 * residue_1 ** 2
        fc = fc + c * rankErr

        G0 = G + c * (U - np.eye(n))
        
        if k1 == 1 or rankErr > tolrank:
            y = np.zeros(k_e)
            for i in range(k_e):
                y[i] = e[i] - G0[I_e[i], J_e[i]]
        
        for itersub in range(1, maxitsub + 1):
            X, y, info = CorMat3Mex(G0, e, I_e, J_e, opts, y)
            P = info['P']
            lambda_ = info['lam']
            rank_X = info['rank']
            major_dualVal = info['dualVal']
            rankErr = abs(np.sum(lambda_) - np.sum(lambda_[:Rank]))
            Totalcall_CN = Totalcall_CN + 1
            subtotalnumb_CG = subtotalnumb_CG + info['numPcg']
            subtotaliter_CN = subtotaliter_CN + info['numIter']
            subtotalnumb_eigendecom = subtotalnumb_eigendecom + info['numEig']
            fc = np.sum(np.sum((X - G) * (X - G)))
            residue_1 = np.sqrt(fc)
            fc = 0.5 * fc + c * rankErr
            
            if itersub <= const_rank_hist:
                rank_hist[itersub - 1] = rank_X
            else:
                rank_hist[:-1] = rank_hist[1:]
                rank_hist[-1] = rank_X
            
            if itersub <= const_funcVal_hist:
                funcVal_hist[itersub - 1] = fc ** 0.5
            else:
                funcVal_hist[:-1] = funcVal_hist[1:]
                funcVal_hist[-1] = fc ** 0.5
            
            if sum_iter + itersub <= const_residue_hist:
                residue_hist[sum_iter + itersub - 1] = residue_1
            else:
                residue_hist[:-1] = residue_hist[1:]
                residue_hist[-1] = residue_1
            
            if rankErr <= tolrank:
                tolsub_check = tolsub_rank
            else:
                tolsub_check = tolsub * max(10, min(100, rank_X / Rank))
            if itersub >= const_funcVal_hist:
                relErr_sub = abs(funcVal_hist[0] - funcVal_hist[-1])
                relErr_sub = relErr_sub / max(residue_cutoff, max(funcVal_hist[0], funcVal_hist[-1]))
            
            if itersub >= const_funcVal_hist and relErr_sub <= tolsub_check:
                break
            elif itersub >= const_rank_hist and abs(rank_hist[0] - rank_hist[-1]) <= progress_rank1 and rank_X - Rank >= progress_rank2:
                break
        
            P1 = P[:, :Rank]
            U = np.dot(P1, P1.T)
            G0 = G + c * (U - np.eye(n))
        
        sum_iter = sum_iter + itersub
        Totalnumb_CG = Totalnumb_CG + subtotalnumb_CG
        Totaliter_CN = Totaliter_CN + subtotaliter_CN
        Totalnumb_eigendecom = Totalnumb_eigendecom + subtotalnumb_eigendecom
        
        if sum_iter >= const_residue_hist:
            relErr = abs(residue_hist[0] - residue_hist[-1])
            relErr = relErr / max(residue_cutoff, max(residue_hist[0], residue_hist[-1]))
        else:
            relErr = abs(residue_hist[0] - residue_hist[sum_iter - 1])
            relErr = relErr / max(residue_cutoff, max(residue_hist[0], residue_hist[sum_iter - 1]))
        
        if relErr <= tolrel:
            if rankErr <= tolrank:
                break
            elif k1 >= const_rankErr_hist and abs(rankErr_hist[0] - rankErr_hist[-1]) <= progress_rankErr:
                finalPCA = 1
                break
        else:
            if abs(relErr_0 - relErr) / max(1, relErr) <= progress_relErr:
                break_level = break_level + 1
                if break_level == 3:
                    if rankErr > tolrank:
                        finalPCA = 1
                    break
        
        k1 = k1 + 1
        relErr_0 = relErr
        
        if rank_X <= Rank:
            c = min(c_max, c)
        else:
            if rankErr / max(1, Rank) > 1.0e-1:
                c = min(c_max, c * alpha_max)
            else:
                c = min(c_max, c * alpha_min)
    
    # check if y is the optimal dual Lagrange multiplier
    X_tmp = G + np.diag(y)
    X_tmp = (X_tmp + X_tmp.T) / 2
    P0, lambda0 = eigh(X_tmp)
    lambda0 = np.real(lambda0)
    if np.all(np.sort(np.abs(lambda0))):
        lambda0 = lambda0[::-1]
    elif np.all(np.sort(np.abs(lambda0[::-1]))):
        pass
    else:
        lambda01, Inx = np.sort(np.abs(lambda0)), np.argsort(np.abs(lambda0))[::-1]
        lambda0 = lambda0[Inx]
    f = np.sum(lambda0[Rank:n] ** 2)
    f = -f + np.dot(y, y)
    f = 0.5 * f
    dual_obj = -f
    
    # final PCA correction
    if len(e_diag) == k_e and finalPCA:
        X = mPCA(P, lambda_, Rank, e_diag)
        rank_X = Rank
        rankErr = 0
        residue_1 = np.sum(np.sum((X - G) * (X - G))) ** 0.5
    infeas = np.zeros(k_e)
    for i in range(k_e):
        infeas[i] = e[i] - X[I_e[i], J_e[i]]
    NormInf = np.linalg.norm(infeas)

    INFOS = {'iter': k1, 'callCN': Totalcall_CN, 'itCN': Totaliter_CN, 'itCG': Totalnumb_CG, 'numEig': Totalnumb_eigendecom, 'rank': rank_X, 'rankErr': rankErr, 'relErr': relErr, 'infeas': NormInf, 'residue': residue_1}
    
    return X, INFOS


# To change the format of time
def time(t):
    t = round(t)
    h = t // 3600
    m = (t % 3600) // 60
    s = t % 60
    return h, m, s


# mexeig decomposition
def MYmexeig(X):
    eigenvalues, eigenvectors = eigh(X)
    P = np.real(eigenvectors)
    lambda_ = np.real(eigenvalues)

    # rearrange lambda in nonincreasing order
    if np.all(np.diff(lambda_) >= 0):
        lambda_ = lambda_[::-1]
        P = P[:, ::-1]
    elif np.all(np.diff(lambda_[::-1]) >= 0):
        return P, lambda_
    else:
        idx = np.argsort(lambda_)[::-1]
        lambda_ = lambda_[idx]
        P = P[:, idx]

    return P, lambda_


def mPCA(P, lambda_, Rank, b=None):
    n = len(lambda_)
    if b is None:
        b = np.ones(n)

    if Rank > 0:
        P1 = P[:, :Rank]
        lambda1 = lambda_[:Rank]
        lambda1 = np.sqrt(lambda1)

        if Rank > 1:
            P1 = P1 @ sp.sparse.diags(lambda1)
        else:
            P1 = P1 * lambda1

        pert_Mat = np.random.rand(n, Rank)
        for i in range(n):
            s = np.linalg.norm(P1[i, :])
            if s < 1.0e-12:  # PCA breakdowns
                P1[i, :] = pert_Mat[i, :]
                s = np.linalg.norm(P1[i, :])
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
        if r > 1:
            lambda1 = np.sqrt(lambda1)
            P1 = P1 @ sp.sparse.diags(lambda1)
            X = P1 @ P1.T
        else:
            X = lambda1 * P1 @ P1.T
    return X


# The InitialPoint function is quite complex and relies on external functions like CorMat3Mex
# which are not provided. A complete translation would require more context and information
# about these external dependencies. Here's a partial translation:

def InitialPoint(G, e, I_e, J_e, Rank, rankErr_CorNewton, X1):
    k_e = len(e)

    maxit = 20
    rank_ratio = 0.90
    tolinfeas = 1.0e-6
    infoNum = {
        'callCN': 0,
        'iterCN': 0,
        'CG': 0,
        'eigendecom': 0
    }

    use_mPCA = 1
    Ind = np.where(I_e == J_e)[0]
    e_diag = e[Ind]

    c0 = 1.0e1
    cmax = 1.0e6
    alpha = 2
    c = c0

    opts = {'disp': 0}
    for iter in range(0, maxit):
        if iter == 0:
            Y = X1
        else:
            if use_mPCA:
                Y = mPCA(P, lambda_, Rank, e_diag)
            else:
                Y = Projr(P, lambda_, Rank)
        infeas = np.zeros((k_e, 1))
        for i in range(0, k_e):
            infeas[i] = e(i) - Y(I_e(i), J_e(i))
        NormInf = np.linalg.norm(infeas, 2)
        if NormInf <= tolinfeas:
            X = Y
            P, lambda_ = MYmexeig(X)
            rank_X = Rank
            rankErr = abs(sum(lambda_) - sum(lambda_[0: Rank]))
            infoNum["eigendecom"] = infoNum["eigendecom"] + 1
            break
        G0 = (G + c * Y) / (1 + c)
        y = np.zeros((k_e, 1))
        for i in range(0, k_e):
            y[i] = e[i] - G0[I_e[i], J_e[i]]

        X, y, info = CorMat3Mex(G0, e, I_e, J_e, opts, y)
        P = info["P"]
        lambda_ = info["lam"]
        rank_X = info["rank"]
        infoNum["callCN"] = infoNum["callCN"] + 1
        infoNum["iterCN"] = infoNum["iterCN"] + info.numIter
        infoNum["CG"] = infoNum["CG"] + info["numPcg"]
        infoNum["eigendecom"] = infoNum["eigendecom"] + info["numEig"]
        rankErr = abs(sum(lambda_) - sum(lambda_[0:Rank]))
        if rankErr <= rank_ratio * max(1, rankErr_CorNewton):
            break
        c = min(alpha * c, cmax)
    return X, P, lambda_, rank_X, rankErr, infoNum

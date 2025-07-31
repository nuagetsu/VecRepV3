'''
Implementing pseudocode from https://proceedings.neurips.cc/paper_files/paper/2002/file/1177967c7957072da3dc1db4ceb30e7a-Paper.pdf to estimate packing dimension.
@inproceedings{inproceedings,
author = {Kégl, Balázs},
year = {2002},
month = {01},
pages = {},
title = {Intrinsic Dimension Estimation Using Packing Numbers}
}
'''

import numpy as np
import math
import random
import statistics

def packing_dim(r_1, r_2, epsilon, testSample, d):
    '''
    :param r_1: Scale resolution r_1
    :param r_2: Scale resolution r_2
    :param epsilon: Epsilon value to dictate accuracy of estimation. Choosing epsilon 0.01 gives 99%.
    :param testSample: Data to compute packing dimension on
    :param d: Distance metric we are computing packing dimension with
    '''
    l = 1
    L_1 = []
    L_2 = []
    while True:
        S_n = np.random.permutation(testSample)
        C = []
        for i in range(len(S_n)):
            badPoint = False
            for j in range(len(C)):
                if (d(S_n[i], C[j]) < r_1):
                    badPoint = True
            if (not badPoint):
                C.append(S_n[i])
        L_1.append(math.log(len(C)))
        C = []
        for i in range(len(testSample)):
            badPoint = False
            for j in range(len(C)):
                if (d(S_n[i], C[j]) < r_2):
                    badPoint = True
            if (not badPoint):
                C.append(S_n[i])
        L_2.append(math.log(len(C)))
        
        num = (np.sum(L_1) / len(L_1)) - (np.sum(L_2) / len(L_2))
        dem = math.log(r_2) - math.log(r_1)
        if (num == 0):
            D_pack = 0
        else:
            D_pack = num / dem
        if (l > 10):
            numerator = math.sqrt(statistics.variance(L_1) + statistics.variance(L_2))
            denom = math.sqrt(l) * (math.log(r_2) - math.log(r_1))
            if (1.65 * (numerator / denom) < D_pack * (1-epsilon) / 2):
                return D_pack
        l += 1

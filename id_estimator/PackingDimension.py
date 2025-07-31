import numpy as np
import math
import random
import statistics

def packing_dim(r_1, r_2, epsilon, testSample, d):
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
                    # print(d(S_n[i], C[j]))
                    badPoint = True
            if (not badPoint):
                # print("not badpoint")
                C.append(S_n[i])
        L_1.append(math.log(len(C)))
        # print(f"L1: {L_1}")
        C = []
        for i in range(len(testSample)):
            badPoint = False
            for j in range(len(C)):
                if (d(S_n[i], C[j]) < r_2):
                    badPoint = True
            if (not badPoint):
                # print("not badpoint")
                C.append(S_n[i])
                # print(f"C: {C}")
        L_2.append(math.log(len(C)))
        # print(f"L2: {L_2}")
        
        num = (np.sum(L_1) / len(L_1)) - (np.sum(L_2) / len(L_2))
        dem = math.log(r_2) - math.log(r_1)
        if (num == 0):
            D_pack = 0
        else:
            D_pack = num / dem
        # D_pack = num / ()
        # print(f"l: {l} and D_pack: {D_pack}")
        if (l > 10):
            # print("l > 10 now")
            numerator = math.sqrt(statistics.variance(L_1) + statistics.variance(L_2))
            denom = math.sqrt(l) * (math.log(r_2) - math.log(r_1))
            # print(numerator / denom)
            if (1.65 * (numerator / denom) < D_pack * (1-epsilon) / 2):
                return D_pack
        l += 1

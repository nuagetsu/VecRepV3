'''
Testing packing dimension ID estimator over different scale resolutions and sample sizes
'''
from torch.utils.data import Subset

import numpy as np
import random


import src.helpers.MetricUtilities as metrics
from src.data_processing.DatasetGetter import get_data, get_data_MStar, get_data_SARDet_100k, get_data_ATRNetSTARAll

from id_estimator.PackingDimension import packing_dim


def test_sample_size(sample_size, r_1, r_2, epsilon):
    list_data = "/home/jovyan/data/ATRNet-STAR_annotations/list_data_all.npz"
    data = np.load(list_data)
    all_data = data["testSample"]
    sample_indices = np.array(random.sample(range(len(all_data)), sample_size))
    testSample = all_data[sample_indices]

    D_pack = packing_dim(r_1, r_2, epsilon, testSample, metrics.dist_fft_numba)
    print(f"D_pack for ATRNET_Star with {sample_size} samples: {D_pack}")

    IMDB_WIKI_data = get_data(128)
    sample_indices = random.sample(range(len(IMDB_WIKI_data)), sample_size)
    sampled_test_data = Subset(IMDB_WIKI_data, sample_indices)
    testSample = np.array(sampled_test_data)

    D_pack = packing_dim(r_1, r_2, epsilon, testSample, metrics.dist_fft_numba)
    print(f"D_pack for IMDB_WIKI with {sample_size} samples: {D_pack}")

    data = get_data_MStar(128)
    sample_indices = random.sample(range(len(data)), sample_size)
    sampled_test_data = Subset(data, sample_indices)
    testSample = np.array([item[0] for item in sampled_test_data])

    D_pack = packing_dim(r_1, r_2, epsilon, testSample, metrics.dist_fft_numba)
    print(f"D_pack for MSTAR with {sample_size} samples: {D_pack}")

    data = get_data_SARDet_100k(128)
    sample_indices = random.sample(range(len(data)), sample_size)
    sampled_test_data = Subset(data, sample_indices)
    testSample = np.array([item[0] for item in sampled_test_data])

    D_pack = packing_dim(r_1, r_2, epsilon, testSample, metrics.dist_fft_numba)
    print(f"D_pack for SARDet_100K with {sample_size} samples: {D_pack}")

def calc_packings_imdb(r1, r2, epsilon, sample_size=100):
    IMDB_WIKI_data = get_data(128)
    sample_indices = random.sample(range(len(IMDB_WIKI_data)), sample_size)
    sampled_test_data = Subset(IMDB_WIKI_data, sample_indices)

    testSample = np.array(sampled_test_data)
    d_pack = packing_dim(r1, r2, epsilon, testSample, metrics.dist_fft_numba)
    print(f"IMDB: {d_pack}")
    return d_pack

def calc_packings_MSTAR(r1,r2,epsilon, sample_size=100):
    data = get_data_MStar(128)
    sample_indices = random.sample(range(len(data)), sample_size)
    sampled_test_data = Subset(data, sample_indices)

    testSample = np.array([item[0] for item in sampled_test_data])

    d_pack = packing_dim(r1, r2, epsilon, testSample, metrics.dist_fft_numba)
    print(f"MSTAR: {d_pack}")

    return d_pack

def calc_packings_SARDET(r1,r2,epsilon, sample_size=100):
    data = get_data_SARDet_100k(128)
    sample_indices = random.sample(range(len(data)), sample_size)
    sampled_test_data = Subset(data, sample_indices)
    testSample = np.array([item[0] for item in sampled_test_data])
    
    d_pack = packing_dim(r1, r2, epsilon, testSample, metrics.dist_fft_numba)
    print(f"SARDET: {d_pack}")

    return d_pack

def calc_packings_ATRNET(r1,r2,epsilon, all_data=[], sample_size=100):
    sample_indices = np.array(random.sample(range(len(all_data)), sample_size))
    testSample = all_data[sample_indices]
    
    d_pack = packing_dim(r1, r2, epsilon, testSample, metrics.dist_fft_numba)
    print(f"ATRNET: {d_pack}")

    return d_pack

def calc_packings(r=[], epsilon=0.01, runs=5, filename="", sample_size = 100):
    avgs_ATRNET = []
    avgs_IMDB = []
    avgs_MSTAR = []
    avgs_SARDET = []

    list_data = "/home/jovyan/data/ATRNet-STAR_annotations/list_data_all.npz"
    data = np.load(list_data)
    all_data = data["testSample"]
    print("done loading all data")

    for i in range(2, len(r)):
        avg_ATRNET = 0
        avg_IMDB = 0
        avg_MSTAR = 0
        avg_SARDET = 0
        print(f"For {r[i-1]} to {r[i]}")
        for j in range(runs):
            
            avg_ATRNET += calc_packings_ATRNET(r[i-1],r[i],epsilon, all_data, sample_size=sample_size)
            avg_IMDB += calc_packings_imdb(r[i-1],r[i],epsilon, sample_size=sample_size)
            avg_MSTAR += calc_packings_MSTAR(r[i-1],r[i],epsilon, sample_size=sample_size)
            avg_SARDET += calc_packings_SARDET(r[i-1],r[i],epsilon, sample_size=sample_size)

        avg_ATRNET = avg_ATRNET / runs
        avg_IMDB = avg_IMDB / runs
        avg_MSTAR = avg_MSTAR / runs
        avg_SARDET = avg_SARDET / runs

        avgs_ATRNET.append(avg_ATRNET)
        avgs_IMDB.append(avg_IMDB)
        avgs_MSTAR.append(avg_MSTAR)
        avgs_SARDET.append(avg_SARDET)
    
    print(f"ATRNET: {avgs_ATRNET}")
    print(f"IMDB: {avgs_IMDB}")
    print(f"MSTAR: {avgs_MSTAR}")
    print(f"SARDET: {avgs_SARDET}")

    # with open(filename, "w") as f:
    #     f.write(f"r: {r}\n")
    #     f.write(f"ATRNET: {avgs_ATRNET}\n")
    #     f.write(f"IMDB: {avgs_IMDB}\n")
    #     f.write(f"MSTAR: {avgs_MSTAR}\n")
    #     f.write(f"SARDET: {avgs_SARDET}")

if __name__ == "__main__":

    # r_1 = 0.2
    # r_2 = 0.1
    # epsilon = 0.5

    # test_sample_size(100)
    # test_sample_size(1000)
    # test_sample_size(2000)
    # test_sample_size(5000)
    # test_sample_size(9000)
    # r = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    r = [0.05, 0.25, 0.3]

    calc_packings(r=r, epsilon=0.01, runs=1, filename="/home/jovyan/evaluation/results/ID_estimator_r_comparisons_2.txt", sample_size=100)



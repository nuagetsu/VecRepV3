'''
Testing for improvements of M-tree search over optimised NCC linear search
'''
from torch.utils.data import Subset

import numpy as np
import random


from src.helpers.MTreeUtilities import getKNearestNeighbours, getMTree, getMTreeFFT, getMTreeFFTNumba
from src.data_processing.DatasetGetter import get_data, get_data_MStar, get_data_SARDet_100k 
import src.data_processing.ImageProducts as ImageProducts

import time

def ncc_pfft_query_time(image_size=128, k=7, runs=100, sample_size=1000, testSample=[], unseen_image=[]):
    avg_ncc_pfft = 0
    time_ncc_pfft = 0
    for j in range(runs):
        start_time = time.perf_counter()
        arr = np.ones(len(testSample))
        unseen_img_arr = ImageProducts.linear_ncc_psearch(testSample, unseen_image, arr)
        imgProd_max_index = np.argpartition(unseen_img_arr, -(k+1))[-(k+1):][1:]
        end_time = time.perf_counter()
        time_ncc_pfft += end_time - start_time
    
    avg_ncc_pfft = time_ncc_pfft / runs
    return avg_ncc_pfft

def ncc_fft_query_time(image_size=128, k=7, runs=100, sample_size=1000, data=[]):
    sample_indices = np.array(random.sample(range(len(data)), sample_size))
    testSample = data[sample_indices]
    unseen_image = data[np.random.randint(len(data))]

    time_ncc_fft = 0
    for j in range(runs):
        start_time = time.perf_counter()
        arr = np.ones(len(testSample))
        unseen_img_arr = ImageProducts.linear_ncc_search(testSample, unseen_image, arr)
        imgProd_max_index = np.argpartition(unseen_img_arr, -(k+1))[-(k+1):][1:]
        end_time = time.perf_counter()
        time_ncc_fft += end_time - start_time
    
    avg_ncc_fft = time_ncc_fft / runs
    return avg_ncc_fft

def mtree_fft_query_time(image_size=128, k=7, runs=100, max_node_size=25, sample_size=1000, data=[]):
    sample_indices = np.array(random.sample(range(len(data)), sample_size))
    testSample = data[sample_indices]
    unseen_image = data[np.random.randint(len(data))]

    time_mtree_fft = 0
    tree_fft = getMTreeFFTNumba(testSample, max_node_size)
    for j in range(runs):
        start_time = time.perf_counter()
        knn = getKNearestNeighbours(tree_fft, unseen_image, k)
        end_time = time.perf_counter()
        time_mtree_fft += end_time - start_time
    
    avg_mtree_fft = time_mtree_fft / runs
    return avg_mtree_fft

def query_percentage_improvements_mtree_ncc(sample_sizes=[], runs=10, k=7):
    print(f"Measuring percentage improvement of mtree over ncc for IMDB, MSTAR, SARDET, ATRNET over {runs} runs for {k} neighbours over variable sample sizes and image size 128")
    list_path = "/home/jovyan/data/ATRNet-STAR_annotations/list_data_all.npz"
    list_path_imdb = "/home/jovyan/data/annotations/imdb_list_data_all_128.npz"
    list_path_mstar = "/home/jovyan/data/annotations/mstar_list_data_all_128.npz"
    list_path_sardet = "/home/jovyan/data/annotations/sardet_list_data_all_128.npz"
    p_incs_imdb = []
    p_incs_MSTAR = []
    p_incs_SARDET = []
    p_incs_ATRNET = []
    IMDB_WIKI_data = np.load(list_path_imdb)
    IMDB_WIKI_data = IMDB_WIKI_data["testSample"]
    MSTAR_data = np.load(list_path_mstar)
    MSTAR_data = MSTAR_data["testSample"]
    SARDET_data = np.load(list_path_sardet)
    SARDET_data = SARDET_data["testSample"]
    ATRNET_data = np.load(list_path)
    all_data = ATRNET_data["testSample"]

    for i in range(len(sample_sizes)):
        print(f"Now doing sample size {sample_sizes[i]}")
        
        # sample_indices = random.sample(range(len(IMDB_WIKI_data)), sample_sizes[i])
        # sampled_test_data = Subset(IMDB_WIKI_data, sample_indices)
        # testSample_imdb = np.array(sampled_test_data)
        # unseen_image_imdb = IMDB_WIKI_data[np.random.randint(len(IMDB_WIKI_data))]

        
        # sample_indices = random.sample(range(len(MSTAR_data)), sample_sizes[i])
        # sampled_test_data = Subset(MSTAR_data, sample_indices)
        # testSample_MSTAR = np.array([item[0] for item in sampled_test_data])
        # unseen_image_MSTAR = MSTAR_data[np.random.randint(len(MSTAR_data))][0]

        
        # sample_indices = random.sample(range(len(SARDET_data)), sample_sizes[i])
        # sampled_test_data = Subset(SARDET_data, sample_indices)
        # testSample_SARDET = np.array([item[0] for item in sampled_test_data])
        # unseen_image_SARDET = SARDET_data[np.random.randint(len(SARDET_data))][0]

        # sample_indices = np.array(random.sample(range(len(all_data)), sample_sizes[i]))
        # testSample_ATRNET = all_data[sample_indices]
        # unseen_image_ATRNET = all_data[np.random.randint(len(all_data))]
    

        avg_ncc_fft_time_imdb = ncc_fft_query_time(image_size=128, k=k, runs=runs, sample_size=sample_sizes[i], data=IMDB_WIKI_data)
        avg_ncc_fft_time_MSTAR = ncc_fft_query_time(image_size=128, k=k, runs=runs, sample_size=sample_sizes[i], data=MSTAR_data)
        avg_ncc_fft_time_SARDET = ncc_fft_query_time(image_size=128, k=k, runs=runs, sample_size=sample_sizes[i], data=SARDET_data)
        avg_ncc_fft_time_ATRNET = ncc_fft_query_time(image_size=128, k=k, runs=runs, sample_size=sample_sizes[i], data=all_data)
        print(f"TIMES NCC: {avg_ncc_fft_time_imdb}, {avg_ncc_fft_time_MSTAR}, {avg_ncc_fft_time_SARDET}, {avg_ncc_fft_time_ATRNET}")

        avg_mtree_fft_time_imdb = mtree_fft_query_time(image_size=128, k=k, runs=runs, max_node_size=39, sample_size=sample_sizes[i], data=IMDB_WIKI_data)
        avg_mtree_fft_time_MSTAR = mtree_fft_query_time(image_size=128, k=k, runs=runs, max_node_size=39, sample_size=sample_sizes[i], data=MSTAR_data)
        avg_mtree_fft_time_SARDET = mtree_fft_query_time(image_size=128, k=k, runs=runs, max_node_size=39, sample_size=sample_sizes[i], data=SARDET_data)
        avg_mtree_fft_time_ATRNET = mtree_fft_query_time(image_size=128, k=k, runs=runs, max_node_size=25, sample_size=sample_sizes[i], data=all_data)
        print(f"TIMES MTREE: {avg_mtree_fft_time_imdb}, {avg_mtree_fft_time_MSTAR}, {avg_mtree_fft_time_SARDET}, {avg_mtree_fft_time_ATRNET}")

        p_inc_imdb = ((avg_ncc_fft_time_imdb - avg_mtree_fft_time_imdb) / avg_mtree_fft_time_imdb) * 100
        p_incs_imdb.append(p_inc_imdb)
        print(f"increase for imdb: {p_inc_imdb}")

        p_inc_MSTAR = ((avg_ncc_fft_time_MSTAR - avg_mtree_fft_time_MSTAR) / avg_mtree_fft_time_MSTAR) * 100
        p_incs_MSTAR.append(p_inc_MSTAR)
        print(f"increase for MSTAR: {p_inc_MSTAR}")

        p_inc_SARDET = ((avg_ncc_fft_time_SARDET - avg_mtree_fft_time_SARDET) / avg_mtree_fft_time_SARDET) * 100
        p_incs_SARDET.append(p_inc_SARDET)
        print(f"increase for SARDET: {p_inc_SARDET}")

        p_inc_ATRNET = ((avg_ncc_fft_time_ATRNET - avg_mtree_fft_time_ATRNET) / avg_mtree_fft_time_ATRNET) * 100
        p_incs_ATRNET.append(p_inc_ATRNET)
        print(f"increase for ATRNET: {p_inc_ATRNET}")
    
    print(f"{p_incs_imdb}")
    print(f"{p_incs_MSTAR}")
    print(f"{p_incs_SARDET}")
    print(f"{p_incs_ATRNET}")

    with open("/home/jovyan/evaluation/results/ID_estimator_comparisons_num3.txt", "w") as file:
        file.write(f"sample_sizes = {sample_sizes}\n")
        file.write(f"p_incs_imdb = {p_incs_imdb}\n")
        file.write(f"p_incs_MSTAR = {p_incs_MSTAR}\n")
        file.write(f"p_incs_SARDET = {p_incs_SARDET}\n")
        file.write(f"p_incs_ATRNET = {p_incs_ATRNET}")

def mtree_ncc_query_times_sample_size(image_size=128, k=7, runs=100, max_node_size=12, list_data="path", sample_sizes = [], filename=""):

    print(f"Query times of diff methods based on kNN with ncc with image size {image_size}, {k} neighbours, mtree max node size {max_node_size} over {runs} runs")

    # data_list = np.load(list_data)
    # data = data_list["testSample"]
    data = get_data(image_size)

            
    avgs_ncc_pfft = []
    avgs_ncc_fft = []
    avgs_ncc_unoptim = []
    avgs_mtree = []
    avgs_mtree_fft = []

    for i in range(len(sample_sizes)):
        
        time_ncc_pfft = 0
        time_ncc_fft = 0
        time_ncc_unoptim = 0
        time_mtree = 0
        time_mtree_fft = 0

        sample_indices = random.sample(range(len(data)), sample_sizes[i])
        sampled_test_data = Subset(data, sample_indices)
        testSample = np.array(sampled_test_data)
        print("done with testSample")
        # tree = getMTree(testSample, max_node_size)
        tree_fft = getMTreeFFTNumba(testSample, max_node_size)
        

        print(f"Now testing with sample size: {sample_sizes[i]}")
        for j in range(runs):

            index1 = np.random.randint(len(data))
            unseen_image = data[index1]

            start_time = time.perf_counter()
            arr = np.ones(len(testSample))
            unseen_img_arr = ImageProducts.linear_ncc_psearch(testSample, unseen_image, arr)
            imgProd_max_index = np.argpartition(unseen_img_arr, -(k+1))[-(k+1):][1:]
            end_time = time.perf_counter()
            time_ncc_pfft += end_time - start_time
            
            start_time = time.perf_counter()
            arr = np.ones(len(testSample))
            unseen_img_arr = ImageProducts.linear_ncc_search(testSample, unseen_image, arr)
            imgProd_max_index = np.argpartition(unseen_img_arr, -(k+1))[-(k+1):][1:]
            end_time = time.perf_counter()
            time_ncc_fft += end_time - start_time

            # start_time = time.perf_counter()
            # arr = []
            # for j in range(len(testSample)):
            #     result = ImageProducts.ncc_scaled(testSample[j], unseen_image)
            #     arr.append(result)
            
            # unseen_img_arr = np.array(arr)
            # imgProd_max_index = np.argpartition(unseen_img_arr, -(k+1))[-(k+1):]
            # time_ncc_unoptim += end_time - start_time

            # start_time = time.perf_counter()
            # knn = getKNearestNeighbours(tree, unseen_image, k)
            # end_time = time.perf_counter()
            # time_mtree += end_time - start_time

            start_time = time.perf_counter()
            knn = getKNearestNeighbours(tree_fft, unseen_image, k)
            end_time = time.perf_counter()
            time_mtree_fft += end_time - start_time
        
        avg_ncc_pfft = time_ncc_pfft / runs
        avgs_ncc_pfft.append(avg_ncc_pfft)
        print(f"Avg time for ncc pfft: {avg_ncc_pfft}")

        avg_ncc_fft = time_ncc_fft / runs
        avgs_ncc_fft.append(avg_ncc_fft)
        print(f"Avg time for ncc fft: {avg_ncc_fft}")

        # avg_ncc_unoptim = time_ncc_unoptim / runs
        # avgs_ncc_unoptim.append(avg_ncc_unoptim)
        # print(f"Avg time for ncc unoptim: {avg_ncc_unoptim}")

        # avg_mtree = time_mtree / runs
        # avgs_mtree.append(avg_mtree)
        # print(f"Avg time for mtree: {avg_mtree}")

        avg_mtree_fft = time_mtree_fft / runs
        avgs_mtree_fft.append(avg_mtree_fft)
        print(f"Avg time for mtree fft: {avg_mtree_fft}")

    print(avgs_ncc_pfft)
    print(avgs_ncc_fft)
    # print(avgs_ncc_unoptim)
    # print(avgs_mtree)
    print(avgs_mtree_fft)
    with open(filename, "w") as file:
        file.write(f"sample_sizes = {sample_sizes}\n")
        file.write(f"avgs_ncc_pfft = {avgs_ncc_pfft}\n")
        file.write(f"avgs_ncc_fft = {avgs_ncc_fft}\n")
        # file.write(f"avgs_ncc_unoptim = {avgs_ncc_unoptim}\n")
        # file.write(f"avgs_mtree = {avgs_mtree}\n")
        file.write(f"avgs_mtree_fft = {avgs_mtree_fft}")

def save_list_data_imdb(image_size):

    data = get_data(image_size)


    testSample = [item for item in data]
    print("Done w testSample")


    filename = "/home/jovyan/data/annotations/imdb_list_data_all_128.npz"
    np.savez(filename, testSample=testSample)

def save_list_data_sardet(image_size):

    data = get_data_SARDet_100k(image_size)


    testSample = [item[0] for item in data]
    print("Done w testSample")


    filename = "/home/jovyan/data/annotations/sardet_list_data_all_128.npz"
    np.savez(filename, testSample=testSample)


if __name__ == "__main__":
    # sample_sizes = [100, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000]
    # save_list_data_imdb(128)
    # save_list_data_sardet(128)
    # query_percentage_improvements_mtree_ncc(sample_sizes=sample_sizes, runs=30, k=7)

    list_path = "/home/jovyan/data/ATRNet-STAR_annotations/list_data_all.npz"
    ATRNET_data = np.load(list_path)
    all_data = ATRNET_data["testSample"]
    avg_ncc_fft_time_ATRNET = ncc_fft_query_time(image_size=128, k=1, runs=30, sample_size=100, data=all_data)
    avg_mtree_fft_time_ATRNET = mtree_fft_query_time(image_size=128, k=1, runs=30, max_node_size=25, sample_size=100, data=all_data)
    print(f"ncc: {avg_ncc_fft_time_ATRNET}")
    print(f"mtree: {avg_mtree_fft_time_ATRNET}")

    # sample_sizes = [1000, 5000, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]
    # mtree_ncc_query_times_sample_size(image_size=128, k=7, runs=30, max_node_size=39, list_data="path", sample_sizes = sample_sizes, filename="/home/jovyan/evaluation/results/IMDB_test_query_sample_sizes.txt")
    
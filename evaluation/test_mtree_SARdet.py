'''
Experiments on the SARdet-100k dataset
'''
from torch.utils.data import Subset
import numpy as np
import random

from src.helpers.MTreeUtilities import getKNearestNeighbours, getMTree, getMTreeFFT, getMTreeFFTNumba
from src.data_processing.DatasetGetter import get_data, get_data_MStar, get_data_SARDet_100k, get_data_ATRNetSTARAll
import src.data_processing.ImageProducts as ImageProducts

import time

def mtree_ncc_query_sample_size(max_node_size=12, image_size=32, k=7, runs=2, sample_sizes=[1]):
    total_time_ncc = 0
    total_time_ncc_parallel = 0
    total_time_ncc_njit = 0
    total_time_mtree_njit = 0
    total_time_mtree = 0
    
    avg_times_ncc = []
    avg_times_ncc_parallel = []
    avg_times_ncc_njit = []
    avg_times_mtree_njit = []
    avg_times_mtree = []

    data = get_data_SARDet_100k(image_size)

    print(f"Average runtime of querying mtrees and nccs for {k} NN over {runs} runs with image size {image_size} and max node size {max_node_size} and variable sample size")
    for i in range(len(sample_sizes)):
        print(f"NOW TRYING sample size: {sample_sizes[i]}")
        total_time_ncc = 0
        total_time_ncc_parallel = 0
        total_time_ncc_njit = 0
        total_time_mtree_njit = 0
        total_time_mtree = 0

        sample_indices = random.sample(range(len(data)), sample_sizes[i])
        sampled_test_data = Subset(data, sample_indices)

        testSample = [item[0] for item in sampled_test_data]
        

        # tree = getMTree(testSample, max_node_size)
        tree_numba = getMTreeFFTNumba(testSample, max_node_size)

        # trans = transforms.Compose([transforms.Resize(img_sizes[i])])
        # t_MNIST_data = trans(MNIST_data)

        # for img in MNIST_data:
        #     img = trans(img)

        for _ in range(runs):
            index1 = np.random.randint(len(data))
            #input1=input_dataset[index1][0].squeeze().to('cpu')
            unseen_image = data[index1][0]

            # start_time = time.perf_counter()
            # arr = []
            # for j in range(len(testSample)):
            #     result = ImageProducts.ncc_scaled(testSample[j], unseen_image)
            #     arr.append(result)
            
            # unseen_img_arr = np.array(arr)
            # #print(unseen_img_arr)
            # imgProd_max_index = np.argpartition(unseen_img_arr, -(k+1))[-(k+1):]
            # end_time = time.perf_counter()

            # total_time_ncc += end_time - start_time

            # start_time = time.perf_counter()
            # arr = np.ones(len(testSample))
            # unseen_img_arr = ImageProducts.linear_ncc_psearch(testSample, unseen_image, arr)
            # imgProd_max_index = np.argpartition(unseen_img_arr, -(k+1))[-(k+1):]
            # end_time = time.perf_counter()

            # total_time_ncc_parallel += end_time - start_time

            start_time = time.perf_counter()
            arr = np.ones(len(testSample))
            unseen_img_arr = ImageProducts.linear_ncc_search(testSample, unseen_image, arr)
            imgProd_max_index = np.argpartition(unseen_img_arr, -(k+1))[-(k+1):]
            end_time = time.perf_counter()

            total_time_ncc_njit += end_time - start_time

            start_time = time.perf_counter()
            imgs = getKNearestNeighbours(tree_numba, unseen_image, k+1)
            end_time = time.perf_counter()
            total_time_mtree_njit += end_time - start_time

            # start_time = time.perf_counter()
            # imgs = getKNearestNeighbours(tree, unseen_image, k+1)
            # end_time = time.perf_counter()
            # total_time_mtree += end_time - start_time

        # avg_ncc = total_time_ncc / runs
        # avg_ncc_parallel = total_time_ncc_parallel / runs
        avg_ncc_njit = total_time_ncc_njit / runs
        avg_mtree_njit = total_time_mtree_njit / runs
        # avg_mtree = total_time_mtree / runs
        
        # avg_times_ncc.append(avg_ncc)
        # avg_times_ncc_parallel.append(avg_ncc_parallel)
        avg_times_ncc_njit.append(avg_ncc_njit)
        avg_times_mtree_njit.append(avg_mtree_njit)
        # avg_times_mtree.append(avg_mtree)
        

        # with open("./test_mtree_query_sample_sizes_avg_times_ncc_.txt", "w") as file:
        #      file.write(f"avg times: {str(avg_times_ncc)}")
        
        # with open("test_mtree_query_sample_sizes_avg_times_mtree_.txt", "w") as file:
        #      file.write(f"avg times: {str(avg_times_mtree)}")

        # print(f"Average runtime of unoptimised ncc search: {avg_ncc:.6f} seconds")
        # print(f"Average runtime of ncc parallel njit search: {avg_ncc_parallel:.6f} seconds")
        print(f"Average runtime of ncc njit search: {avg_ncc_njit:.6f} seconds")
        print(f"Average runtime of mtree njit search: {avg_mtree_njit:.6f} seconds")
        # print(f"Average runtime of unoptimised mtree search: {avg_mtree:.6f} seconds")
    
    # print(avg_times_ncc)
    # print(avg_times_ncc_parallel)
    print(avg_times_ncc_njit)
    print(avg_times_mtree_njit)
    # print(avg_times_mtree)

if __name__ == "__main__":
    sample_sizes = [1000, 5000, 10000, 20000, 50000, 100000]
    #image_sizes = [32, 64, 128, 140, 160, 180, 200]
    # mtree_ncc_query_sample_size(max_node_size=12, image_size=16, k=7, runs=3, sample_sizes=sample_sizes)
    # mtree_init_sample_size(max_node_size=12, image_size=16, k=7, runs=2, sample_sizes=sample_sizes)
    # mtree_init_img_size(max_node_size=12, image_sizes=image_sizes, k=7, runs=2, sample_size=5000)
    # matG_init_sample_size(image_size=16, k=7, runs=1, sample_sizes=[100, 500, 1000, 5000])
    # matG_init_img_size(image_sizes=image_sizes, k=7, runs=2, sample_size=50)
    #mtree_ncc_query_image_size(max_node_size=12, image_sizes=image_sizes, k=7, runs=100, sample_size=500)
    # sample_sizes = [4000]
    mtree_ncc_query_sample_size(max_node_size=39, image_size=128, k=7, runs=30, sample_sizes=sample_sizes)
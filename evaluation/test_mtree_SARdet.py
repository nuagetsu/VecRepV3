import sys
import os
path = os.path.abspath("../")
sys.path.append(path)

import torch
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import numpy as np
import numba as nb
from numba import prange
import math
import pandas as pd
import random

from line_profiler import profile

import src.helpers.MetricUtilities as metrics
import src.data_processing.ImageProducts as ImageProducts


from mtree.mtree import MTree
import mtree.mtree as mtree

import glob
from PIL import Image
import cv2

import time

import json
def getKNearestNeighbours(tree, point, k):
    l = tree.search(point, k)
    imgs = list(l)
    return imgs

def getMTree(data, k):
    # k: desired number of nearest neighbours
    tree = MTree(metrics.distance, max_node_size=k)
    tree.add_all(data)
    return tree

def getMTreeFFT(data, k):
    # k: desired number of nearest neighbours
    tree = MTree(metrics.dist_fft, max_node_size=k)
    tree.add_all(data)
    return tree

def getMTreeFFTNumba(data, k):
    # k: desired number of nearest neighbours
    tree = MTree(metrics.dist_fft_numba, max_node_size=k)
    tree.add_all(data)
    return tree


@nb.njit(parallel=True, cache=True)
def linear_ncc_psearch(testSample, unseen_image, arr):
    for i in prange(len(testSample)):
        arr[i] = ImageProducts.ncc_fft_numba(testSample[i], unseen_image)

    return arr

@nb.njit(cache=True)
def linear_ncc_search(testSample, unseen_image, arr):
    for i in range(len(testSample)):
        arr[i] = ImageProducts.ncc_fft_numba(testSample[i], unseen_image)

    return arr



# TODO look at train/val/test.json format and extract out the category id and the id to class mapping at the v end.
class CustomDatasetSARDet_100k(Dataset):
    def __init__(self, transform=None):
        self.transform = transform

        self.imgs_path = "/home/jovyan/data/SARDet_100k/SARDet_100K/JPEGImages/"
        file_list = glob.glob(self.imgs_path + "*")
        self.data = []

        # with open("../data/SARDet_100k/SARDet_100K/mapping.json") as annotations:
        #     mappings = json.load(annotations)

        for dir_path in file_list:
            for img_path in glob.glob(dir_path + "/*.png"):
                # self.data.append([img_path, mappings[img_path.split("/")[-1]]])
                self.data.append([img_path, "yippee"])
            for img_path in glob.glob(dir_path + "/*.jpg"):
                # self.data.append([img_path, mappings[img_path.split("/")[-1]]])
                self.data.append([img_path, "yippee"])
        

    # Defining the length of the dataset
    def __len__(self):
        return len(self.data)

    # Defining the method to get an item from the dataset
    def __getitem__(self, index):
        data_path = self.data[index]
        image = Image.open(data_path[0])
        image = transforms.functional.to_grayscale(image)
        # class_id = self.class_map[data_path[1]]
        class_id = data_path[1]
        # Applying the transform
        if self.transform:
            image = self.transform(image)
        
        return image.squeeze().to('cpu').numpy(), class_id

def get_data_SARDet_100k(size):
    transform = transforms.Compose([
    transforms.Resize((size, size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
    ])
    return CustomDatasetSARDet_100k(transform)


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
        

        tree = getMTree(testSample, max_node_size)
        tree_numba = getMTreeFFTNumba(testSample, max_node_size)

        # trans = transforms.Compose([transforms.Resize(img_sizes[i])])
        # t_MNIST_data = trans(MNIST_data)

        # for img in MNIST_data:
        #     img = trans(img)

        for _ in range(runs):
            index1 = np.random.randint(len(data))
            #input1=input_dataset[index1][0].squeeze().to('cpu')
            unseen_image = data[index1][0]

            start_time = time.perf_counter()
            arr = []
            for j in range(len(testSample)):
                result = ImageProducts.ncc_scaled(testSample[j], unseen_image)
                arr.append(result)
            
            unseen_img_arr = np.array(arr)
            #print(unseen_img_arr)
            imgProd_max_index = np.argpartition(unseen_img_arr, -(k+1))[-(k+1):]
            end_time = time.perf_counter()

            total_time_ncc += end_time - start_time

            start_time = time.perf_counter()
            arr = np.ones(len(testSample))
            unseen_img_arr = linear_ncc_psearch(testSample, unseen_image, arr)
            imgProd_max_index = np.argpartition(unseen_img_arr, -(k+1))[-(k+1):]
            end_time = time.perf_counter()

            total_time_ncc_parallel += end_time - start_time

            start_time = time.perf_counter()
            arr = np.ones(len(testSample))
            unseen_img_arr = linear_ncc_search(testSample, unseen_image, arr)
            imgProd_max_index = np.argpartition(unseen_img_arr, -(k+1))[-(k+1):]
            end_time = time.perf_counter()

            total_time_ncc_njit += end_time - start_time

            start_time = time.perf_counter()
            imgs = getKNearestNeighbours(tree_numba, unseen_image, k+1)
            end_time = time.perf_counter()
            total_time_mtree_njit += end_time - start_time

            start_time = time.perf_counter()
            imgs = getKNearestNeighbours(tree, unseen_image, k+1)
            end_time = time.perf_counter()
            total_time_mtree += end_time - start_time

        avg_ncc = total_time_ncc / runs
        avg_ncc_parallel = total_time_ncc_parallel / runs
        avg_ncc_njit = total_time_ncc_njit / runs
        avg_mtree_njit = total_time_mtree_njit / runs
        avg_mtree = total_time_mtree / runs
        
        avg_times_ncc.append(avg_ncc)
        avg_times_ncc_parallel.append(avg_ncc_parallel)
        avg_times_ncc_njit.append(avg_ncc_njit)
        avg_times_mtree_njit.append(avg_mtree_njit)
        avg_times_mtree.append(avg_mtree)
        

        # with open("./test_mtree_query_sample_sizes_avg_times_ncc_.txt", "w") as file:
        #      file.write(f"avg times: {str(avg_times_ncc)}")
        
        # with open("test_mtree_query_sample_sizes_avg_times_mtree_.txt", "w") as file:
        #      file.write(f"avg times: {str(avg_times_mtree)}")

        print(f"Average runtime of unoptimised ncc search: {avg_ncc:.6f} seconds")
        print(f"Average runtime of ncc parallel njit search: {avg_ncc_parallel:.6f} seconds")
        print(f"Average runtime of ncc njit search: {avg_ncc_njit:.6f} seconds")
        print(f"Average runtime of mtree njit search: {avg_mtree_njit:.6f} seconds")
        print(f"Average runtime of unoptimised mtree search: {avg_mtree:.6f} seconds")
    
    print(avg_times_ncc)
    print(avg_times_ncc_parallel)
    print(avg_times_ncc_njit)
    print(avg_times_mtree_njit)
    print(avg_times_mtree)



    

if __name__ == "__main__":
    #sample_sizes = [1000, 5000, 10000, 20000, 50000, 100000, 200000, 300000, 400000, 500000]
    #image_sizes = [32, 64, 128, 140, 160, 180, 200]
    # mtree_ncc_query_sample_size(max_node_size=12, image_size=16, k=7, runs=3, sample_sizes=sample_sizes)
    # mtree_init_sample_size(max_node_size=12, image_size=16, k=7, runs=2, sample_sizes=sample_sizes)
    # mtree_init_img_size(max_node_size=12, image_sizes=image_sizes, k=7, runs=2, sample_size=5000)
    # matG_init_sample_size(image_size=16, k=7, runs=1, sample_sizes=[100, 500, 1000, 5000])
    # matG_init_img_size(image_sizes=image_sizes, k=7, runs=2, sample_size=50)
    #mtree_ncc_query_image_size(max_node_size=12, image_sizes=image_sizes, k=7, runs=100, sample_size=500)
    sample_sizes = [1000, 5000, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]
    mtree_ncc_query_sample_size(max_node_size=15, image_size=200, k=7, runs=1, sample_sizes=sample_sizes)
import sys
import os
path = os.path.abspath("../")
sys.path.append(path)
print(path)

import torch
from torch.utils.data import DataLoader, TensorDataset, Sampler, random_split, Dataset, Subset
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
import random


import src.helpers.MetricUtilities as metrics
import src.data_processing.ImageProducts as ImageProducts

from mtree.mtree import MTree
import mtree.mtree as mtree

import glob
from PIL import Image

import time
import json
import xml.etree.ElementTree as ET


class CustomDatasetATRNetSTARAll(Dataset):
    def __init__(self, filename, transform=None):
        data_list = np.load(filename)
        self.data = data_list["data"]
        self.transform = transform
        

    # Defining the length of the dataset
    def __len__(self):
        return len(self.data)

    # Defining the method to get an item from the dataset
    def __getitem__(self, index):
        data_path = str(self.data[index])
        image = Image.open(data_path)
        image = transforms.functional.to_grayscale(image)
        # Applying the transform
        if self.transform:
            image = self.transform(image)
        
        return image.squeeze().to('cpu').numpy()

def get_data_ATRNetSTARAll(size, filename):
    transform = transforms.Compose([
    transforms.Resize((size, size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
    ])
    return CustomDatasetATRNetSTARAll(filename, transform)

def getKNearestNeighbours(tree, point, k):
    l = tree.search(point, k)
    imgs = list(l)
    return imgs

def getMTree(data, k, promote=mtree.M_LB_DIST_confirmed, partition=mtree.generalized_hyperplane, d=metrics.distance):
    # k: desired number of nearest neighbours
    tree = MTree(d, max_node_size=k, promote=promote, partition=partition)
    tree.add_all(data)
    return tree

def getMTreeFFT(data, k):
    # k: desired number of nearest neighbours
    tree = MTree(metrics.dist_fft, max_node_size=k)
    tree.add_all(data)
    return tree

def getMTreeFFTNumba(data, k, promote=mtree.M_LB_DIST_confirmed, partition=mtree.generalized_hyperplane):
    # k: desired number of nearest neighbours
    tree = MTree(metrics.dist_fft_numba, max_node_size=k, promote=promote, partition=partition)
    tree.add_all(data)
    return tree


def mtree_ncc_query_times_sample_size(image_size=128, k=7, runs=100, max_node_size=12, list_data="path", sample_sizes = []):

    print(f"Query times of diff methods based on kNN with ncc with image size {image_size}, {k} neighbours, mtree max node size {max_node_size} over {runs} runs")

    data_list = np.load(list_data)
    data = data_list["testSample"]

            
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
        testSample = data[sample_indices]
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
    with open("/home/jovyan/evaluation/results/test_query_sample_sizes.txt", "w") as file:
        file.write(f"sample_sizes = {sample_sizes}\n")
        file.write(f"avgs_ncc_pfft = {avgs_ncc_pfft}\n")
        file.write(f"avgs_ncc_fft = {avgs_ncc_fft}\n")
        # file.write(f"avgs_ncc_unoptim = {avgs_ncc_unoptim}\n")
        # file.write(f"avgs_mtree = {avgs_mtree}\n")
        file.write(f"avgs_mtree_fft = {avgs_mtree_fft}")

def mtree_init_times_sample_sizes(image_size=128, k=7, runs=100, max_node_size=12, list_data="path", sample_sizes = []):

    print(f"Init times of mtree w {image_size}, {k} neighbours, max_node_size {max_node_size}, variable sample size over {runs} runs")

    data_list = np.load(list_data)
    data = data_list["testSample"]

    # avgs_mtree = []
    avgs_mtree_fft = []

    for i in range(len(sample_sizes)):
    
        # time_mtree = 0
        time_mtree_fft = 0
        
        print(f"Now testing with sample size: {sample_sizes[i]}")
        for j in range(runs):
            sample_indices = random.sample(range(len(data)), sample_sizes[i])
            testSample = data[sample_indices]

            # start_time = time.perf_counter()
            # tree = getMTree(testSample, max_node_size)
            # end_time = time.perf_counter()
            # time_mtree += end_time - start_time

            start_time = time.perf_counter()
            tree_fft = getMTreeFFTNumba(testSample, max_node_size)
            end_time = time.perf_counter()
            time_mtree_fft += end_time - start_time
    
        # avg_mtree = time_mtree / runs
        # avgs_mtree.append(avg_mtree)
        # print(f"Avg time for mtree: {avgs_mtree}")

        avg_mtree_fft = time_mtree_fft / runs
        avgs_mtree_fft.append(avg_mtree_fft)
        print(f"Avg time for mtree fft: {avg_mtree_fft}")

    # print(avgs_mtree)
    print(avgs_mtree_fft)
    with open("/home/jovyan/evaluation/results/test_init_max_sample_sizes.txt", "w") as file:
        # file.write(f"avgs_mtree = {avgs_mtree}\n")
        file.write(f"avgs_mtree_fft = {avgs_mtree_fft}")

def mtree_init_times_max_node_size(image_size=128, k=7, runs=100, max_node_sizes=[], list_data="path", sample_size = 1000):

    print(f"Init times of mtree w {image_size}, {k} neighbours, variable mtree max node size, sample size {sample_size} over {runs} runs")

    data_list = np.load(list_data)
    data = data_list["testSample"]

    # avgs_mtree = []
    avgs_mtree_fft = []

    for i in range(len(max_node_sizes)):
    
        time_mtree_fft = 0
        
        print(f"Now testing with max node size: {max_node_sizes[i]}")
        for j in range(runs):
            sample_indices = random.sample(range(len(data)), sample_size)
            testSample = data[sample_indices]

            start_time = time.perf_counter()
            tree_fft = getMTreeFFTNumba(testSample, max_node_sizes[i])
            end_time = time.perf_counter()
            time_mtree_fft += end_time - start_time
    

        avg_mtree_fft = time_mtree_fft / runs
        avgs_mtree_fft.append(avg_mtree_fft)
        print(f"Avg time for mtree fft: {avg_mtree_fft}")

    print(avgs_mtree_fft)
    with open("/home/jovyan/evaluation/results/test_init_max_node_sizes.txt", "w") as file:
        file.write(f"avgs_mtree_fft = {avgs_mtree_fft}")

def ncc_unoptim_query_times_sample_size(image_size=128, k=7, runs=100, max_node_size=12, list_data="path", sample_sizes = []):

    print(f"Query times of ncc unoptimised based on kNN with ncc with image size {image_size}, {k} neighbours, mtree max node size {max_node_size} over {runs} runs")

    data_list = np.load(list_data)
    data = data_list["testSample"]

    avgs_ncc_unoptim = []

    for i in range(len(sample_sizes)):
        
        time_ncc_unoptim = 0

        sample_indices = random.sample(range(len(data)), sample_sizes[i])
        testSample = data[sample_indices]
        print("done with testSample")
        

        print(f"Now testing with sample size: {sample_sizes[i]}")
        for j in range(runs):

            index1 = np.random.randint(len(data))
            unseen_image = data[index1]

            start_time = time.perf_counter()
            arr = []
            for j in range(len(testSample)):
                result = ImageProducts.ncc_scaled(testSample[j], unseen_image)
                arr.append(result)
            
            unseen_img_arr = np.array(arr)
            imgProd_max_index = np.argpartition(unseen_img_arr, -(k+1))[-(k+1):]
            end_time = time.perf_counter()
            
            time_ncc_unoptim += end_time - start_time

        avg_ncc_unoptim = time_ncc_unoptim / runs
        avgs_ncc_unoptim.append(avg_ncc_unoptim)
        print(f"Avg time for ncc unoptim: {avg_ncc_unoptim}")

    print(avgs_ncc_unoptim)

    with open("/home/jovyan/evaluation/results/test_ncc_unoptim_query_sample_sizes.txt", "w") as file:
        file.write(f"sample_sizes = {sample_sizes}\n")
        file.write(f"avgs_ncc_unoptim = {avgs_ncc_unoptim}")

def mtree_ncc_query_times_image_size(image_sizes=[], k=7, runs=30, max_node_size=25, split_path="", sample_size = 1000):

    print(f"Query times of diff methods based on kNN with ncc with variable image size, {k} neighbours, mtree max node size {max_node_size} over {runs} runs")

            
    avgs_ncc_pfft = []
    avgs_ncc_fft = []
    avgs_ncc_unoptim = []
    avgs_mtree = []
    avgs_mtree_fft = []

    for i in range(len(image_sizes)):

        dataset = get_data_ATRNetSTARAll(image_sizes[i], split_path)
        data = np.array([item for item in dataset])
    
        
        time_ncc_pfft = 0
        time_ncc_fft = 0
        time_ncc_unoptim = 0
        time_mtree = 0
        time_mtree_fft = 0

        sample_indices = random.sample(range(len(data)), sample_size)
        testSample = data[sample_indices]
        print("done with testSample")
        # tree = getMTree(testSample, max_node_size)
        tree_fft = getMTreeFFTNumba(testSample, max_node_size)
        

        print(f"Now testing with image size: {image_sizes[i]}")
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
    with open("/home/jovyan/evaluation/results/test_query_image_sizes_2.txt", "w") as file:
        file.write(f"image_sizes = {image_sizes}\n")
        file.write(f"avgs_ncc_pfft = {avgs_ncc_pfft}\n")
        file.write(f"avgs_ncc_fft = {avgs_ncc_fft}\n")
        # file.write(f"avgs_ncc_unoptim = {avgs_ncc_unoptim}\n")
        # file.write(f"avgs_mtree = {avgs_mtree}\n")
        file.write(f"avgs_mtree_fft = {avgs_mtree_fft}")

def mtree_init_times_image_size(image_sizes=[], k=7, runs=30, max_node_size=25, split_path="", sample_size = 1000):

    print(f"Init times of mtree based on kNN with ncc with variable image size, {k} neighbours, mtree max node size {max_node_size} over {runs} runs")

            
    avgs_mtree_fft = []


    for i in range(len(image_sizes)):

        dataset = get_data_ATRNetSTARAll(image_sizes[i], split_path)
        data = np.array([item for item in dataset])
    
        time_mtree_fft = 0

        sample_indices = random.sample(range(len(data)), sample_size)
        testSample = data[sample_indices]
        print("done with testSample")
        

        print(f"Now testing with image size: {image_sizes[i]}")
        for j in range(runs):
            start_time = time.perf_counter()
            tree_fft = getMTreeFFTNumba(testSample, max_node_size)
            end_time = time.perf_counter()
            time_mtree_fft += end_time - start_time

        avg_mtree_fft = time_mtree_fft / runs
        avgs_mtree_fft.append(avg_mtree_fft)
        print(f"Avg time for mtree fft: {avg_mtree_fft}")

    print(avgs_mtree_fft)

    with open("/home/jovyan/evaluation/results/test_init_image_sizes_2.txt", "w") as file:
        file.write(f"image_sizes = {image_sizes}\n")
        file.write(f"avgs_mtree_fft = {avgs_mtree_fft}")

def mtree_unoptim_query_times_sample_size(image_size=128, k=7, runs=30, max_node_size=25, list_data="path", sample_sizes = []):

    print(f"Query times of ncc unoptimised based on kNN with ncc with image size {image_size}, {k} neighbours, mtree max node size {max_node_size} over {runs} runs")

    data_list = np.load(list_data)
    data = data_list["testSample"]

    avgs_mtree_unoptim = []

    for i in range(len(sample_sizes)):
        
        time_mtree_unoptim = 0

        sample_indices = random.sample(range(len(data)), sample_sizes[i])
        testSample = data[sample_indices]
        tree = getMTree(testSample, k=k)
        print("done with testSample")
        

        print(f"Now testing with sample size: {sample_sizes[i]}")
        for j in range(runs):

            index1 = np.random.randint(len(data))
            unseen_image = data[index1]

            start_time = time.perf_counter()
            neighbours = getKNearestNeighbours(tree, unseen_image, k=k)
            end_time = time.perf_counter()
            
            time_mtree_unoptim += end_time - start_time

        avg_mtree_unoptim = time_mtree_unoptim / runs
        avgs_mtree_unoptim.append(avg_mtree_unoptim)
        print(f"Avg time for ncc unoptim: {avg_mtree_unoptim}")

    print(avgs_mtree_unoptim)

    with open("/home/jovyan/evaluation/results/test_mtree_unoptim_query_sample_sizes.txt", "w") as file:
        file.write(f"sample_sizes = {sample_sizes}\n")
        file.write(f"avgs_ncc_unoptim = {avgs_mtree_unoptim}")

if __name__ == "__main__":
    list_data = "/home/jovyan/data/ATRNet-STAR_annotations/list_data_all.npz"
    split_path = "/home/jovyan/data/ATRNet-STAR_annotations/all_filepaths.npz"
    sample_sizes = [1000, 5000, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]
    image_sizes= [140, 160, 180, 200]
    max_node_sizes = list(np.arange(41))[5:]
    k = 7
    image_size = 128
    max_node_size = 25

    # mtree_unoptim_query_times_sample_size(image_size=image_size, k=k, runs=2, max_node_size=max_node_size, list_data=list_data, sample_sizes = sample_sizes)

    # mtree_ncc_query_times_sample_size(image_size=image_size, k=k, runs=100, max_node_size=max_node_size, list_data=list_data, sample_sizes = sample_sizes)
    # ncc_unoptim_query_times_sample_size(image_size=image_size, k=k, runs=1, max_node_size=max_node_sizes, list_data=list_data, sample_sizes = sample_sizes)

    mtree_init_times_sample_sizes(image_size=image_size, k=k, runs=1, max_node_size=max_node_size, list_data=list_data, sample_sizes = [80000])
    # mtree_init_times_max_node_size(image_size=image_size, k=k, runs=2, max_node_sizes=max_node_sizes, list_data=list_data, sample_size = 1000)

    # mtree_ncc_query_times_image_size(image_sizes=image_sizes, k=7, runs=30, max_node_size=25, split_path=split_path, sample_size = 1000)
    # mtree_init_times_image_size(image_sizes=[128], k=7, runs=30, max_node_size=25, split_path=split_path, sample_size = 1000)

    # mtree_ncc_query_times_image_size(image_sizes=image_sizes, k=k, runs=100, max_node_size=max_node_sizes, split_path=split_path, sample_size = 1000)
    # mtree_init_times_image_size(image_sizes=image_sizes, k=7, runs=2, max_node_size=max_node_sizes, split_path=split_path, sample_size = 1000)

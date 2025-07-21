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

import statistics


class CustomDataset(Dataset):
    def __init__(self, transform=None):
        self.transform = transform

        self.imgs_path = "/home/jovyan/data/imdb_wiki/"
        file_list = glob.glob(self.imgs_path + "*")
        self.images = []
        for class_path in file_list:
            for dir_path in glob.glob(class_path + "/*"):
                for img_path in glob.glob(dir_path + "/*.jpg"):
                    self.images.append(img_path)

    # Defining the length of the dataset
    def __len__(self):
        return len(self.images)

    # Defining the method to get an item from the dataset
    def __getitem__(self, index):
        image_path = self.images[index]
        image = Image.open(image_path)
        image = transforms.functional.to_grayscale(image)

        # Applying the transform
        if self.transform:
            image = self.transform(image)
        
        return image.squeeze().to('cpu').numpy()

def get_data(size):
    transform = transforms.Compose([
    transforms.Resize((size, size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
    ])
    return CustomDataset(transform)


class CustomDatasetMStar(Dataset):
    def __init__(self, transform=None):
        self.transform = transform

        self.imgs_path = "/home/jovyan/data/mstar/Padded_imgs/"
        file_list = glob.glob(self.imgs_path + "*")
        self.data = []
        for class_path in file_list:
            class_name = class_path.split("/")[-1]
            for img_path in glob.glob(class_path + "/*.JPG"):
                self.data.append([img_path, class_name])
        #print(self.data)
        self.class_map = {"2S1" : 0, "BRDM_2": 1, "BTR_60": 2, "D7": 3, "SLICY": 4, "T62": 5, "ZIL131": 6, "ZSU_23_4": 7}

    # Defining the length of the dataset
    def __len__(self):
        return len(self.data)

    # Defining the method to get an item from the dataset
    def __getitem__(self, index):
        data_path = self.data[index]
        image = Image.open(data_path[0])
        image = transforms.functional.to_grayscale(image)
        class_id = self.class_map[data_path[1]]
        # Applying the transform
        if self.transform:
            image = self.transform(image)
        
        return image.squeeze().to('cpu').numpy(), class_id
    
def get_data_MStar(size):
    transform = transforms.Compose([
    transforms.Resize((size, size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
    ])
    return CustomDatasetMStar(transform)

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

def test_sample_size(sample_size):
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


def calc_packings_imdb(r1, r2, epsilon):
    IMDB_WIKI_data = get_data(128)
    sample_indices = random.sample(range(len(IMDB_WIKI_data)), 100)
    sampled_test_data = Subset(IMDB_WIKI_data, sample_indices)

    testSample = np.array(sampled_test_data)
    d_pack = packing_dim(r1, r2, epsilon, testSample, metrics.dist_fft_numba)
    print(f"IMDB: {d_pack}")
    return d_pack

def calc_packings_MSTAR(r1,r2,epsilon):
    data = get_data_MStar(128)
    sample_indices = random.sample(range(len(data)), 100)
    sampled_test_data = Subset(data, sample_indices)

    testSample = np.array([item[0] for item in sampled_test_data])

    d_pack = packing_dim(r1, r2, epsilon, testSample, metrics.dist_fft_numba)
    print(f"MSTAR: {d_pack}")

    return d_pack

def calc_packings_SARDET(r1,r2,epsilon):
    data = get_data_SARDet_100k(128)
    sample_indices = random.sample(range(len(data)), 100)
    sampled_test_data = Subset(data, sample_indices)
    testSample = np.array([item[0] for item in sampled_test_data])
    
    d_pack = packing_dim(r1, r2, epsilon, testSample, metrics.dist_fft_numba)
    print(f"SARDET: {d_pack}")

    return d_pack

def calc_packings_ATRNET(r1,r2,epsilon, all_data=[]):
    sample_indices = np.array(random.sample(range(len(all_data)), 100))
    testSample = all_data[sample_indices]
    
    d_pack = packing_dim(r1, r2, epsilon, testSample, metrics.dist_fft_numba)
    print(f"ATRNET: {d_pack}")

    return d_pack

def calc_packings(r=[], epsilon=0.01, runs=5, filename=""):
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
            
            avg_ATRNET += calc_packings_ATRNET(r[i-1],r[i],epsilon, all_data)
            avg_IMDB += calc_packings_imdb(r[i-1],r[i],epsilon)
            avg_MSTAR += calc_packings_MSTAR(r[i-1],r[i],epsilon)
            avg_SARDET += calc_packings_SARDET(r[i-1],r[i],epsilon)

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

    with open(filename, "w") as f:
        f.write(f"r: {r}\n")
        f.write(f"ATRNET: {avgs_ATRNET}\n")
        f.write(f"IMDB: {avgs_IMDB}\n")
        f.write(f"MSTAR: {avgs_MSTAR}\n")
        f.write(f"SARDET: {avgs_SARDET}")

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

    calc_packings(r=r, epsilon=0.01, runs=5, filename="/home/jovyan/evaluation/results/ID_estimator_r_comparisons_2.txt")



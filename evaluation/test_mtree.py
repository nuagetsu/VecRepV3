import sys
import os
path = os.path.abspath("../")
sys.path.append(path)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, TensorDataset, Sampler, random_split, Dataset, Subset
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
import random

from collections import defaultdict, Counter
from line_profiler import profile
from scipy.linalg import orthogonal_procrustes

import src.data_processing.ImageCalculations as imgcalc
import src.visualization.ImagePlots as imgplt
import src.helpers.ModelUtilities as models
import src.helpers.TCNNModelUtilities as tcnnmodels
import src.data_processing.Utilities as utils
import src.helpers.FilepathUtils as Futils
import src.data_processing.EmbeddingFunctions as embedfunc
import src.data_processing.ImageProducts as ImageProducts
import src.helpers.MetricUtilities as metrics

from src.visualization import BFmethod
from functools import partial
from learnable_polyphase_sampling.learn_poly_sampling.layers import get_logits_model, PolyphaseInvariantDown2D, LPS
from learnable_polyphase_sampling.learn_poly_sampling.layers.polydown import set_pool

from mtree.mtree import MTree
import mtree.mtree as mtree

import glob
from PIL import Image

import time


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO: Put these in a separate file for MTreeUtils...
def getKNearestNeighbours(tree, point, k):
    l = tree.search(point, k)
    imgs = list(l)
    return imgs

def getMTree(data, k):
    # k: desired number of nearest neighbours
    tree = MTree(metrics.distance, max_node_size=k)
    tree.add_all(data)
    return tree

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])   

class CustomDataset(Dataset):
    def __init__(self, transform=None):
        self.transform = transform

        self.imgs_path = "/home/jovyan/data/imdb_wiki/"
        file_list = glob.glob(self.imgs_path + "*")
        #print(file_list)
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

def get_data(size):
    transform = transforms.Compose([
    transforms.Resize((size, size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
    ])
    return CustomDataset(transform)

def get_data_MStar(size):
    transform = transforms.Compose([
    transforms.Resize((size, size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
    ])
    return CustomDatasetMStar(transform)


def plot_data_mtree_ncc(x_axis="", title="", filename="", varied_arr=[], data1=[], data2=[], max_node_size=12):
    data_ncc = []
    data_mtree = []
    for i in range(len(varied_arr)):
        data_ncc.append([varied_arr[i], data1[i]])
        data_mtree.append([varied_arr[i], data2[i]])

    # print(data_ncc)
    # print(data_mtree)
    data_ncc = np.array(data_ncc)
    data_mtree = np.array(data_mtree)
    x_ncc, y_ncc = data_ncc.T
    x_mtree, y_mtree = data_mtree.T

    plt.plot(x_ncc, y_ncc, label = "ncc search", linestyle="-")
    plt.plot(x_mtree, y_mtree, label = f"mtree search, node size {max_node_size}", linestyle="--")
    plt.legend()
    plt.xlabel(x_axis)  # Title for the x-axis
    plt.ylabel("Runtime (s)")  # Title for the y-axis
    plt.title(title)
    # note its over 5 runs
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', dpi=1000)

def plot_data_mtree_init(x_axis="", title="", filename="", varied_arr=[], data=[], max_node_size=12):
    data_mtree_init = []
    for i in range(len(varied_arr)):
        data_mtree_init.append([varied_arr[i], data[i]])

    data_mtree_init = np.array(data_mtree_init)

    x_mtree_init, y_mtree_init = data_mtree_init.T

    plt.plot(x_mtree_init, y_mtree_init, label = f"mtree init", linestyle="-")
    plt.legend()
    plt.xlabel(x_axis)
    plt.ylabel("Runtime (s)")
    plt.title(title)
    # note its over 3 runs
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', dpi=1000)

def plot_data_matG_init(x_axis="", title="", filename="", varied_arr=[], data=[]):
    data_init = []
    for i in range(len(varied_arr)):
        data_init.append([varied_arr[i], data[i]])

    data_init = np.array(data_init)

    x, y = data_init.T

    plt.plot(x, y, label = f"matG init", linestyle="-")
    plt.legend()
    plt.xlabel(x_axis)
    plt.ylabel("Runtime (s)")
    plt.title(title)
    # note its over 3 runs
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', dpi=1000)

def mtree_ncc_query_sample_size(max_node_size=12, image_size=32, k=7, runs=2, sample_sizes=[1]):
    total_time_ncc = 0
    total_time_mtree = 0
    avg_times_ncc = []
    avg_times_mtree = []

    IMDB_WIKI_data = get_data(image_size)

    print(f"Average runtime of querying mtree and ncc for {k} NN over {runs} runs with image size {image_size} and max node size {max_node_size} and variable sample size")
    for i in range(len(sample_sizes)):
        print(f"NOW TRYING sample size: {sample_sizes[i]}")
        total_time_ncc = 0
        total_time_mtree = 0

        sample_indices = random.sample(range(len(IMDB_WIKI_data)), sample_sizes[i])
        sampled_test_data = Subset(IMDB_WIKI_data, sample_indices)

        testSample = sampled_test_data      
        tree = getMTree(testSample, max_node_size)

        for _ in range(runs):
            index1 = np.random.randint(len(IMDB_WIKI_data))
            unseen_image = IMDB_WIKI_data[index1]

            start_time = time.time()
            arr = []
            for j in range(len(testSample)):
                result = ImageProducts.ncc_scaled(testSample[j], unseen_image)
                arr.append(result)
            
            unseen_img_arr = np.array(arr)
            imgProd_max_index = np.argpartition(unseen_img_arr, -(k+1))[-(k+1):]
            end_time = time.time()

            total_time_ncc += end_time - start_time

            start_time = time.time()
            imgs = getKNearestNeighbours(tree, unseen_image, k+1)
            end_time = time.time()
            total_time_mtree += end_time - start_time

        avg_ncc = total_time_ncc / runs
        avg_mtree = total_time_mtree / runs
        avg_times_ncc.append(avg_ncc)
        avg_times_mtree.append(avg_mtree)

        with open("test_mtree_query_sample_sizes_avg_times_ncc_.txt", "w") as file:
            file.write(f"avg times: {str(avg_times_ncc)}")
        
        with open("test_mtree_query_sample_sizes_avg_times_mtree_.txt", "w") as file:
            file.write(f"avg times: {str(avg_times_mtree)}")

        print(f"Average runtime of ncc search: {avg_ncc:.6f} seconds")
        print(f"Average runtime of mtree search: {avg_mtree:.6f} seconds")
    plot_data_mtree_ncc(x_axis="Sample size", title=f"Average runtime of finding {k} neighbours with image size {image_size} against sample sizes", filename="test_mtree_query_sample_sizes.png", 
                        varied_arr=sample_sizes, data1=avg_times_ncc, data2=avg_times_mtree, max_node_size=max_node_size)

def mtree_ncc_query_sample_size_mstar(max_node_size=12, image_size=32, k=7, runs=2, sample_sizes=[1]):
    total_time_ncc = 0
    total_time_mtree = 0
    avg_times_ncc = []
    avg_times_mtree = []

    data = get_data_MStar(image_size)

    print(f"Average runtime of querying mtree and ncc for {k} NN over {runs} runs with image size {image_size} and max node size {max_node_size} and variable sample size")
    for i in range(len(sample_sizes)):
        print(f"NOW TRYING sample size: {sample_sizes[i]}")
        total_time_ncc = 0
        total_time_mtree = 0
        sample_indices = random.sample(range(len(data)), sample_sizes[i])
        sampled_test_data = Subset(data, sample_indices)

        testSample = [item[0] for item in sampled_test_data]
        

        tree = getMTree(testSample, max_node_size)

        # trans = transforms.Compose([transforms.Resize(img_sizes[i])])
        # t_MNIST_data = trans(MNIST_data)

        # for img in MNIST_data:
        #     img = trans(img)

        for _ in range(runs):
            index1 = np.random.randint(len(data))
            #input1=input_dataset[index1][0].squeeze().to('cpu')
            unseen_image = data[index1][0]

            start_time = time.time()
            arr = []
            for j in range(len(testSample)):
                result = ImageProducts.ncc_scaled(testSample[j], unseen_image)
                arr.append(result)
            
            unseen_img_arr = np.array(arr)
            #print(unseen_img_arr)
            imgProd_max_index = np.argpartition(unseen_img_arr, -(k+1))[-(k+1):]
            end_time = time.time()

            total_time_ncc += end_time - start_time

            start_time = time.time()
            imgs = getKNearestNeighbours(tree, unseen_image, k+1)
            end_time = time.time()
            total_time_mtree += end_time - start_time

        avg_ncc = total_time_ncc / runs
        avg_mtree = total_time_mtree / runs
        avg_times_ncc.append(avg_ncc)
        avg_times_mtree.append(avg_mtree)

        # with open("./test_mtree_query_sample_sizes_avg_times_ncc_.txt", "w") as file:
        #      file.write(f"avg times: {str(avg_times_ncc)}")
        
        # with open("test_mtree_query_sample_sizes_avg_times_mtree_.txt", "w") as file:
        #      file.write(f"avg times: {str(avg_times_mtree)}")

        print(f"Average runtime of ncc search: {avg_ncc:.6f} seconds")
        print(f"Average runtime of mtree search: {avg_mtree:.6f} seconds")
    # plot_data_mtree_ncc(x_axis="Sample size", title=f"Average runtime of finding {k} neighbours with image size {image_size} against sample sizes", filename="test_mtree_query_sample_sizes.png", 
    #                     varied_arr=sample_sizes, data1=avg_times_ncc, data2=avg_times_mtree, max_node_size=max_node_size)


def mtree_query_sample_size(max_node_size=12, image_size=16, k=7, runs=100, sample_sizes=[500000]):
    total_time_mtree = 0
    avg_times_mtree = []

    IMDB_WIKI_data = get_data(image_size)

    print(f"Average runtime of querying mtree for {k} NN over {runs} runs with image size {image_size} and max node size {max_node_size} and variable sample size")
    for i in range(len(sample_sizes)):
        print(f"NOW TRYING sample size: {sample_sizes[i]}")
        total_time_mtree = 0

        sample_indices = random.sample(range(len(IMDB_WIKI_data)), sample_sizes[i])
        sampled_test_data = Subset(IMDB_WIKI_data, sample_indices)

        testSample = sampled_test_data      
        tree = getMTree(testSample, max_node_size)

        for _ in range(runs):
            index1 = np.random.randint(len(IMDB_WIKI_data))
            unseen_image = IMDB_WIKI_data[index1]

            start_time = time.time()
            imgs = getKNearestNeighbours(tree, unseen_image, k+1)
            end_time = time.time()
            total_time_mtree += end_time - start_time

        avg_mtree = total_time_mtree / runs
        avg_times_mtree.append(avg_mtree)

        print(f"Average runtime of mtree search: {avg_mtree:.6f} seconds")
    

def mtree_ncc_query_image_size(max_node_size=12, image_sizes=[], k=7, runs=2, sample_size=50):
    total_time_ncc = 0
    total_time_mtree = 0
    avg_times_ncc = []
    avg_times_mtree = []

    

    print(f"Average runtime of querying mtree and ncc for {k} NN over {runs} runs with sample size {sample_size} and max node size {max_node_size} and variable image")
    for i in range(len(image_sizes)):
        print(f"NOW TRYING image size: {image_sizes[i]}")
        total_time_ncc = 0
        total_time_mtree = 0

        IMDB_WIKI_data = get_data(image_sizes[i])

        sample_indices = random.sample(range(len(IMDB_WIKI_data)), sample_size)
        sampled_test_data = Subset(IMDB_WIKI_data, sample_indices)

        testSample = sampled_test_data      
        tree = getMTree(testSample, max_node_size)

        for _ in range(runs):
            index1 = np.random.randint(len(IMDB_WIKI_data))
            unseen_image = IMDB_WIKI_data[index1]

            start_time = time.time()
            arr = []
            for j in range(len(testSample)):
                result = ImageProducts.ncc_scaled(testSample[j], unseen_image)
                arr.append(result)
            
            unseen_img_arr = np.array(arr)
            imgProd_max_index = np.argpartition(unseen_img_arr, -(k+1))[-(k+1):]
            end_time = time.time()

            total_time_ncc += end_time - start_time

            start_time = time.time()
            imgs = getKNearestNeighbours(tree, unseen_image, k+1)
            end_time = time.time()
            total_time_mtree += end_time - start_time

        avg_ncc = total_time_ncc / runs
        avg_mtree = total_time_mtree / runs
        avg_times_ncc.append(avg_ncc)
        avg_times_mtree.append(avg_mtree)

        with open("test_mtree_query_image_sizes_avg_times_ncc_.txt", "w") as file:
            file.write(f"avg times: {str(avg_times_ncc)}")
        
        with open("test_mtree_query_image_sizes_avg_times_mtree_.txt", "w") as file:
            file.write(f"avg times: {str(avg_times_mtree)}")

        print(f"Average runtime of ncc search: {avg_ncc:.6f} seconds")
        print(f"Average runtime of mtree search: {avg_mtree:.6f} seconds")
    plot_data_mtree_ncc(x_axis="Image size", title=f"Average runtime of finding {k} neighbours with sample size {sample_size} against image sizes", filename="test_mtree_query_image_sizes.png", 
                        varied_arr=image_sizes, data1=avg_times_ncc, data2=avg_times_mtree, max_node_size=max_node_size)



# Get the array out into a file
def mtree_init_sample_size(max_node_size=12, image_size=32, k=7, runs=2, sample_sizes=[1]):
    total_time_mtree_init = 0
    avg_times_mtree_init = []

    IMDB_WIKI_data = get_data(image_size)

    print(f"Average runtime of initialising mtree over {runs} runs with max node size {max_node_size} and image size {image_size} and variable sample sizes")

    for i in range(len(sample_sizes)):
        print(f"NOW TRYING sample size: {sample_sizes[i]}")
        total_time_mtree_init = 0

        sample_indices = random.sample(range(len(IMDB_WIKI_data)), sample_sizes[i])
        sampled_test_data = Subset(IMDB_WIKI_data, sample_indices)

        testSample = sampled_test_data

        for _ in range(runs):
            start_time = time.time()
            tree = getMTree(testSample, max_node_size)
            end_time = time.time()
            total_time_mtree_init += end_time - start_time
            print(f"Finished one run of mtree init...")

        
        avg_mtree_init = total_time_mtree_init / runs
        avg_times_mtree_init.append(avg_mtree_init)

        
        with open("test_mtree_init_sample_sizes_avg_times_mtree_init.txt", "w") as file:
            file.write(f"avg times: {str(avg_times_mtree_init)}")

        print(f"Average runtime of mtree init: {avg_mtree_init:.6f} seconds")
    
    plot_data_mtree_init(x_axis="Sample size", title=f"Average runtime of initialising mtree with image size {image_size}, max node size {max_node_size} against sample sizes", filename="test_mtree_init_sample_sizes.png", 
                        varied_arr=sample_sizes, data=avg_times_mtree_init, max_node_size=max_node_size)

def mtree_init_sample_size_mstar(max_node_size=12, image_size=32, k=7, runs=2, sample_sizes=[1]):
    total_time_mtree_init = 0
    avg_times_mtree_init = []

    data = get_data_MStar(image_size)

    print(f"Average runtime of initialising mtree over {runs} runs with max node size {max_node_size} and image size {image_size} and variable sample sizes")

    for i in range(len(sample_sizes)):
        print(f"NOW TRYING sample size: {sample_sizes[i]}")
        total_time_mtree_init = 0

        sample_indices = random.sample(range(len(data)), sample_sizes[i])
        sampled_test_data = Subset(data, sample_indices)

        testSample = [item[0] for item in sampled_test_data]
        #sampled_test_data[:,0]

        for _ in range(runs):
            start_time = time.time()
            tree = getMTree(testSample, max_node_size)
            end_time = time.time()
            total_time_mtree_init += end_time - start_time
            print(f"Finished one run of mtree init...")

        
        avg_mtree_init = total_time_mtree_init / runs
        avg_times_mtree_init.append(avg_mtree_init)

        
        # with open("test_mtree_init_sample_sizes_avg_times_mtree_init.txt", "w") as file:
        #     file.write(f"avg times: {str(avg_times_mtree_init)}")

        print(f"Average runtime of mtree init: {avg_mtree_init:.6f} seconds")
    
    # plot_data_mtree_init(x_axis="Sample size", title=f"Average runtime of initialising mtree with image size {image_size}, max node size {max_node_size} against sample sizes", filename="test_mtree_init_sample_sizes.png", 
    #                     varied_arr=sample_sizes, data=avg_times_mtree_init, max_node_size=max_node_size)
    

# Get the array out into a file. Test out to make sure file writing works first.
def mtree_init_img_size(max_node_size=12, image_sizes=[], k=7, runs=2, sample_size=5000):
    total_time_mtree_init = 0
    avg_times_mtree_init = []


    print(f"Average runtime of initialising mtree over {runs} runs with variable image sizes and sample size {sample_size} and max nodes size {max_node_size} ")

    for i in range(len(image_sizes)):
        IMDB_WIKI_data = get_data(image_sizes[i])
        print(f"NOW TRYING image size: {image_sizes[i]}")
        total_time_mtree_init = 0

        sample_indices = random.sample(range(len(IMDB_WIKI_data)), sample_size)
        sampled_test_data = Subset(IMDB_WIKI_data, sample_indices)

        testSample = sampled_test_data

        for _ in range(runs):
            start_time = time.time()
            tree = getMTree(testSample, max_node_size)
            end_time = time.time()
            total_time_mtree_init += end_time - start_time
            print(f"Finished one run of mtree init...")

        
        avg_mtree_init = total_time_mtree_init / runs
        avg_times_mtree_init.append(avg_mtree_init)

        
        with open("test_mtree_init_image_sizes_avg_times_mtree_init.txt", "w") as file:
            file.write(f"avg times: {str(avg_times_mtree_init)}")

        print(f"Average runtime of mtree init: {avg_mtree_init:.6f} seconds")
    
    plot_data_mtree_init(x_axis="Image size", title=f"Average runtime of initialising mtree with sample size {sample_size} and max node size {max_node_size} against image sizes", filename="test_mtree_init_image_sizes_max_node_12.png", 
                        varied_arr=image_sizes, data=avg_times_mtree_init, max_node_size=max_node_size)

# decide how far I can take the initialisation of matrix G though... run it the last
def matG_init_sample_size(image_size=32, k=7, runs=2, sample_sizes=[1]):
    total_time_matG_init = 0
    avg_times_matG_init = []

    IMDB_WIKI_data = get_data(image_size)

    print(f"Average runtime of initialising matG over {runs} runs with image size {image_size} and variable sample sizes")

    for i in range(len(sample_sizes)):
        print(f"NOW TRYING sample size: {sample_sizes[i]}")
        total_time_matG_init = 0

        sample_indices = random.sample(range(len(IMDB_WIKI_data)), sample_sizes[i])
        sampled_test_data = Subset(IMDB_WIKI_data, sample_indices)

        testSample = sampled_test_data

        for _ in range(runs):
            start_time = time.time()
            matG = imgcalc.get_matrixG(testSample, "ncc_scaled_-1")
            end_time = time.time()
            total_time_matG_init += end_time - start_time
            print(f"Finished one run of matG init...")

        
        avg_matG_init = total_time_matG_init / runs
        avg_times_matG_init.append(avg_matG_init)

        
        with open("test_matG_init_sample_sizes_avg_times_matG_init.txt", "w") as file:
            file.write(f"avg times: {str(avg_times_matG_init)}")

        print(f"Average runtime of matG init: {avg_matG_init:.6f} seconds")
    
    plot_data_matG_init(x_axis="Sample size", title=f"Average runtime of initialising mat G with image size {image_size}, against sample sizes", filename="test_matG_init_sample_sizes.png", 
                        varied_arr=sample_sizes, data=avg_times_matG_init)
    

def matG_init_img_size(image_sizes=[], k=7, runs=2, sample_size=5000):
    total_time_matG_init = 0
    avg_times_matG_init = []

    

    print(f"Average runtime of initialising matG over {runs} runs with sample size {sample_size} and variable image sizes")

    for i in range(len(image_sizes)):
        print(f"NOW TRYING image size: {image_sizes[i]}")
        total_time_matG_init = 0

        IMDB_WIKI_data = get_data(image_sizes[i])
        sample_indices = random.sample(range(len(IMDB_WIKI_data)), sample_size)
        sampled_test_data = Subset(IMDB_WIKI_data, sample_indices)

        testSample = sampled_test_data

        for _ in range(runs):
            start_time = time.time()
            matG = imgcalc.get_matrixG(testSample, "ncc_scaled_-1")
            end_time = time.time()
            total_time_matG_init += end_time - start_time
            print(f"Finished one run of matG init...")

        
        avg_matG_init = total_time_matG_init / runs
        avg_times_matG_init.append(avg_matG_init)

        
        with open("test_matG_init_image_sizes_avg_times_matG_init.txt", "w") as file:
            file.write(f"avg times: {str(avg_times_matG_init)}")

        print(f"Average runtime of matG init: {avg_matG_init:.6f} seconds")
    
    plot_data_matG_init(x_axis="Image size", title=f"Average runtime of initialising mat G with sample size {sample_size}, against image sizes", filename="test_matG_init_image_sizes.png", 
                        varied_arr=image_sizes, data=avg_times_matG_init)
    


#def mtree_ncc_query_k(max_node_size=12, image_size=32, ks=[], runs=2, sample_size=5000):
if __name__ == "__main__":
    #sample_sizes = [1000, 5000, 10000, 20000, 50000, 100000, 200000, 300000, 400000, 500000]
    #image_sizes = [32, 64, 128, 140, 160, 180, 200]
    # mtree_ncc_query_sample_size(max_node_size=12, image_size=16, k=7, runs=3, sample_sizes=sample_sizes)
    # mtree_init_sample_size(max_node_size=12, image_size=16, k=7, runs=2, sample_sizes=sample_sizes)
    # mtree_init_img_size(max_node_size=12, image_sizes=image_sizes, k=7, runs=2, sample_size=5000)
    # matG_init_sample_size(image_size=16, k=7, runs=1, sample_sizes=[100, 500, 1000, 5000])
    # matG_init_img_size(image_sizes=image_sizes, k=7, runs=2, sample_size=50)
    #mtree_ncc_query_image_size(max_node_size=12, image_sizes=image_sizes, k=7, runs=100, sample_size=500)
    mtree_ncc_query_sample_size_mstar(max_node_size=15, image_size=200, k=7, runs=10, sample_sizes=[100, 1000, 2000, 5000, 7000, 9465])
    mtree_init_sample_size_mstar(max_node_size=15, image_size=200, k=7, runs=2, sample_sizes=[100, 1000, 2000, 5000, 9465])









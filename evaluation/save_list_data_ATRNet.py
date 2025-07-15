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



class CustomDatasetATRNetSTARSplit(Dataset):
    def __init__(self, filename, annotations_path, transform=None):
        data = np.load(filename)
        self.build = data["build"]
        self.test = data["test"]
        self.transform = transform
        with open(annotations_path) as f:
            self.annotations = json.load(f)
        

    # Defining the length of the dataset
    def __len__(self):
        return len(self.build)
    
    def get_test_len(self):
        return len(self.test)

    # Defining the method to get an item from the dataset
    def __getitem__(self, index):
        data_path = str(self.build[index])
        image = Image.open(data_path)
        image = transforms.functional.to_grayscale(image)
        filename = data_path.split("/")[-1]
        broad_class = self.annotations[filename]["class"]
        subclass = self.annotations[filename]["subclass"]
        subclass_type = self.annotations[filename]["type"]
        # Applying the transform
        if self.transform:
            image = self.transform(image)
        
        return image.squeeze().to('cpu').numpy(), broad_class, subclass, subclass_type
    
    def get_test(self, index):
        data_path = self.test[index]
        image = Image.open(data_path)
        image = transforms.functional.to_grayscale(image)
        filename = data_path.split("/")[-1]
        broad_class = self.annotations[filename]["class"]
        subclass = self.annotations[filename]["subclass"]
        subclass_type = self.annotations[filename]["type"]
        # Applying the transform
        if self.transform:
            image = self.transform(image)
        
        return image.squeeze().to('cpu').numpy(), broad_class, subclass, subclass_type


def get_data_ATRNetSTARSplit(size, filename, annotations_path):
    transform = transforms.Compose([
    transforms.Resize((size, size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
    ])
    return CustomDatasetATRNetSTARSplit(filename, annotations_path, transform)

def save_list_data(image_size, split, annotations_path):

    data = get_data_ATRNetSTARSplit(image_size, split, annotations_path)


    testSample = [item[0] for item in data]
    print("Done w testSample")
    class_names = [item[1] for item in data]
    print("Done w class_names")
    subclass_names = [item[2] for item in data]
    print("Done w subclass_names")
    type_names = [item[3] for item in data]

    split_dict = {}
    split_dict["testSample"] = testSample
    split_dict["class_names"] = class_names
    split_dict["subclass_names"] = subclass_names
    split_dict["type_names"] = type_names

    filename = "/home/jovyan/data/ATRNet-STAR_annotations/list_data_" + split.split("/")[-1]
    np.savez(filename, testSample=testSample, class_names=class_names, subclass_names=subclass_names, type_names=type_names)


if __name__ == "__main__":
    annotations_path = "/home/jovyan/data/ATRNet-STAR_annotations/SOC_40classes_annotations.json"
    image_size = 128
    # save_list_data(image_size, "/home/jovyan/data/ATRNet-STAR_annotations/SOC_40classes_100_0_split.npz", annotations_path)
    # save_list_data(image_size, "/home/jovyan/data/ATRNet-STAR_annotations/SOC_40classes_90_10_split.npz", annotations_path)
    save_list_data(image_size, "/home/jovyan/data/ATRNet-STAR_annotations/SOC_40classes_80_20_split.npz", annotations_path)
    # save_list_data(image_size, "/home/jovyan/data/ATRNet-STAR_annotations/SOC_40classes_70_30_split.npz", annotations_path)
    # save_list_data(image_size, "/home/jovyan/data/ATRNet-STAR_annotations/SOC_40classes_60_40_split.npz", annotations_path)
    # save_list_data(image_size, "/home/jovyan/data/ATRNet-STAR_annotations/SOC_40classes_50_50_split.npz", annotations_path)
    # save_list_data(image_size, "/home/jovyan/data/ATRNet-STAR_annotations/SOC_40classes_40_60_split.npz", annotations_path)
    # save_list_data(image_size, "/home/jovyan/data/ATRNet-STAR_annotations/SOC_40classes_30_70_split.npz", annotations_path)
    # save_list_data(image_size, "/home/jovyan/data/ATRNet-STAR_annotations/SOC_40classes_20_80_split.npz", annotations_path)
    # save_list_data(image_size, "/home/jovyan/data/ATRNet-STAR_annotations/SOC_40classes_10_90_split.npz", annotations_path)
    
    
    
    
    
    
    

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


def save_as_dict_keys_all(imgs_path):
    #imgs_path = "/home/jovyan/data/ATRNet-STAR/EOC_azimuth/"
    #file_list = [imgs_path + "test_60/"]
    # file_list = [glob.glob(annotation_path + "*.json")]
    #print(file_list)    
    annotations = {}
    for xml_file_path in glob.glob(imgs_path + "/*.xml"):
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        img_path = root.find('filename').text

        obj = root.find('object')
        broad_class = obj.find('class').text
        subclass = obj.find('subclass').text
        subclass_type = "_".join(obj.find('type').text.split("_")[1:])

        xmin = int(obj.find('xmin').text)
        ymin = int(obj.find('ymin').text)
        x_len = int(obj.find('xmax').text) - xmin
        y_len = int(obj.find('ymax').text) - ymin
        bbox = [int(xmin), int(ymin), x_len, y_len]

        scene_name = root.find('scene').find('scene_name').text
        
        sensor = root.find('sensor')
        platform = sensor.find('platform').text
        strimap = sensor.find('imaging_mode').text
        band = sensor.find('band').text
        polarization = sensor.find('polarization').text
        range_resolution = sensor.find('range_resolution').text[:-1]
        cross_range_resolution = sensor.find('cross_range_resolution').text[:-1]
        depression_angle = sensor.find('depression_angle').text[:-1]
        target_azimuth_angle = sensor.find('target_azimuth_angle').text[:-1]

        annotations[img_path] = {
            "class": broad_class,
            "subclass": subclass,
            "type": subclass_type,
            "bbox": bbox,
            "scene_name": scene_name,
            "platform": platform,
            "strimap": strimap,
            "band": band,
            "polarization": polarization,
            "range_resolution": range_resolution,
            "cross_range_resolution": cross_range_resolution,
            "depression_angle": depression_angle,
            "target_azimuth_angle": target_azimuth_angle
        }


    # with open('/home/jovyan/data/ATRNet-STAR_annotations.json', 'w', encoding='utf-8') as f:
    #     json.dump(annotations, f, ensure_ascii=False, indent=4)
    file_path = "/home/jovyan/data/ATRNet-STAR_annotations/all_annotations.json"
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(annotations, f, ensure_ascii=False, indent=4)


def sort_by_class_all(imgs_path):
    # file_list = glob.glob(imgs_path + "*/")
    # print(file_list)


    type_names = ["Excelle_GT", "GL8", "CS75_Plus", "Starlight_4500", "Cheetah_CFA6473C", "8228-5", "Arrizo 5", "qq3", "Blazer_1998", "HOWO", "Duolika", "EQ6608LTV", "Forthing_Lingzhi",
            "Tianjin_DFH2200B", "Tianjin_KR230", "J6P", "Jiabao_T51", "BJ1045V9JB5-54", "Wall_poer", "Wall_Voleex_C50", "EV160B", "CA7180A3E", "h5", "N1", "HLF25_II", "Junling", "Patriot", 
            "SY5033XJH", "MKC", "Proud_2009", "V80", "Outlander_2003", "ZL40F", "DeLong_M3000", "DeLong_X3000", "Aochi_1800", "Aochi_Hongrui", "Rongguang_V", "YZK6590XCA", "ZK6120HY1"]
    categories = {}
    for i in range(len(type_names)):
        categories[type_names[i]] = []

    for xml_file_path in glob.glob(imgs_path + "/*.xml"):
        print(xml_file_path)
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        img_path = root.find('filename').text
        img_path = imgs_path + img_path
        subclass_type = "_".join(root.find('object').find('type').text.split("_")[1:])
        print(subclass_type)
        categories[subclass_type] = categories[subclass_type] + [img_path]
        # self.data.append(chip_data)

    file_path = "/home/jovyan/data/ATRNet-STAR_annotations/all_byclass.json"
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(categories, f, ensure_ascii=False, indent=4)

def get_new_dict_data_all(imgs_path):
    save_as_dict_keys_all(imgs_path)
    sort_by_class_all(imgs_path)

    
def split_data(categories, p_left, p_right, filename):
    data = np.array([])
    test = np.array([])
    for key in categories:
        entry = np.array("/home/jovyan/data/ATRNet-STAR/all/" + key)
        # print(entry)
        # print(type(entry))
        left_len = math.floor(p_left * len(entry))
        right_len = math.ceil(p_right * len(entry))
        sample_indices = np.array(random.sample(range(len(entry)), left_len))
        # print(sample_indices)
        # sampled_test_data = Subset(data, sample_indices)

        mask = np.ones(len(entry), np.bool)
        mask[sample_indices] = 0
        # These are all wrong omg ded
        left_partition = entry[sample_indices]
        right_partition = entry[mask]
        # print(type(left_partition))
        # print(left_partition)
        # print(right_partition)
        data = np.concatenate((data, left_partition))
        test = np.concatenate((test, right_partition), axis=0)
        # other_indices = np.arange(len(entry))[mask]
        # left_partition = Subset(entry, sample_indices)
        # right_partition = Subset(entry, )
        # other_data = data[mask]
    
    np.savez(filename, build=data, test=test)
    return filename

def get_data_all(annotations, filename):
    data = np.array([])
    for key in annotations:
        entry = np.array("/home/jovyan/data/ATRNet-STAR/all/" + key)
        data = np.append(data, entry)

    
    np.savez(filename, data=data)
    return filename

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


def save_list_data(image_size, split):

    data = get_data_ATRNetSTARAll(image_size, split)


    testSample = [item for item in data]
    print("Done w testSample")


    filename = "/home/jovyan/data/ATRNet-STAR_annotations/list_data_all.npz"
    np.savez(filename, testSample=testSample)
    # with open("/home/jovyan/data/ATRNet-STAR_annotations/" + filename, 'w', encoding='utf-8') as f:
    #     json.dump(split_dict, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    with open("/home/jovyan/data/ATRNet-STAR_annotations/all_annotations.json") as f:
        annotations = json.load(f)

    save_list_data(128, "/home/jovyan/data/ATRNet-STAR_annotations/all_filepaths.npz")
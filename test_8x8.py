import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Sampler, random_split, Dataset
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import numpy as np
import math
from collections import defaultdict
from line_profiler import profile
from sklearn.model_selection import train_test_split

import src.data_processing.BruteForceEstimator as bfEstimator
import src.visualization.BFmethod as graphing
import src.visualization.Metrics as metrics
import src.data_processing.ImageProducts as ImageProducts

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

class SimpleCNN(nn.Module):
    def __init__(self, dimensions=128): 
        super(SimpleCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)  
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1) 
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)  
        self.bn5 = nn.BatchNorm2d(512)
        self.relu = nn.LeakyReLU(0.1)
        self.fc1 = nn.Linear(512 * 8 * 8, 512)  
        self.fc2 = nn.Linear(512, dimensions)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        
        residual = x
        x = x + residual  

        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)
        x = torch.flatten(x, 1)
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.normalize(x, p=2, dim=1)

        return x
    
# ----------------------------------Image Input----------------------------------
IMAGE_TYPES = ["NbinMmax_ones", "Nbin", "triangles", "triangle_mean_subtracted"]

IMAGE_FILTERS = ["unique", "Nmax_ones", "one_island"]

IMAGE_PRODUCT_TYPES = ["ncc", "ncc_scaled"]

EMBEDDING_TYPES = ["pencorr_D"]

dimensions = 128

imageType = "triangles"
filters = ["100max_ones"]
imageProductType = "ncc_scaled_-1"
overwrite = {"imgSet": False, "imgProd": False, "embedding": False}
weight = None
embeddingType = f"pencorr_{dimensions}"
k=3

def loss_fn(A, G):
    return torch.norm(A - G, p='fro')  # Frobenius norm

bruteForceEstimator = bfEstimator.BruteForceEstimator(
    imageType=imageType, filters=filters, imageProductType=imageProductType, embeddingType=embeddingType, overwrite=overwrite)

index1 = np.random.randint(len(bruteForceEstimator.imageSet))
index2 = np.random.randint(len(bruteForceEstimator.imageSet))

print("index 1: ", index1)
print("index 2: ", index2)

input1=bruteForceEstimator.imageSet[index1]
input2=bruteForceEstimator.imageSet[index2]

print("image 1: ", input1)
print("image 2: ", input2)

model = SimpleCNN().cuda()
model.load_state_dict(torch.load('model/best_model_batch_greyscale.pth'))
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

scale = ImageProducts.scale_min(ImageProducts.ncc, -1)
NCC_scaled_value = scale(input1,input2)
print("\nCalculated values")
print("\nscaled NCC: ",NCC_scaled_value)

input_dataset = []
input_images = [input1, input2] 
for i in range(len(input_images)):
    img = np.array(input_images[i], dtype=np.float64)
    img = torch.from_numpy(img)
    img = img.unsqueeze(0).unsqueeze(0).cuda().double()
    # img = img.repeat(1, 3, 1, 1)
    # print("size of img: ", img.shape)
    input_dataset.append(img)

stacked_tensor = torch.stack(input_dataset)
input_dataset = stacked_tensor.cpu().numpy()      
input_dataset = [torch.tensor(data).cuda().float() for data in input_dataset]
#----------------------Metric 1 - Difference in Values-----------------
embedded_vector_image1 = model(input_dataset[0])
embedded_vector_image2 = model(input_dataset[1])

# print("embedded vector for image 1: ", embedded_vector_image1)
# print("embedded vector for image 2: ", embedded_vector_image2)

dot_product_value = torch.sum(embedded_vector_image1 * embedded_vector_image2, dim=1) 

print("dot product value of model: ", dot_product_value.item())

NCC_scaled_value = torch.tensor(NCC_scaled_value).to(dot_product_value.device).float()
if NCC_scaled_value.ndim == 0:
    NCC_scaled_value = NCC_scaled_value.unsqueeze(0)

train_loss_value = loss_fn(dot_product_value, NCC_scaled_value) 
print("loss: ", train_loss_value.item())

print("\nIn comparison with Pencorr method")

matrixA = bruteForceEstimator.matrixA
matrixG = bruteForceEstimator.matrixG
dot_product_matrix = np.dot(matrixA.T, matrixA)
dot_product_value_Pencorr = dot_product_matrix[index1][index2]
difference = abs(dot_product_value_Pencorr - dot_product_value)

print("\ndot product value of pencorr: ", dot_product_value_Pencorr)
print("absolute difference in values: ", difference.item())

#----------------------Metric 2 - KNNIoU-----------------
print(f"\nBrute Force Method -- KNN-IOU score")
kscores = []
vectorc=[]
vectorb = bruteForceEstimator.matrixG[index1]
for j in range(len(matrixA[0])):
    vectorc.append(np.dot(bruteForceEstimator.matrixA[:, j], bruteForceEstimator.matrixA[:, index1]))
kscore= metrics.get_k_neighbour_score(vectorb, vectorc, k)
kscores.append(kscore)
print(f"Estimating K-Score for Image {index1}: K-Score = {kscore}")

kscores = []
vectorc=[]
vectorb = bruteForceEstimator.matrixG[index2]
for j in range(len(matrixA[0])):
    vectorc.append(np.dot(bruteForceEstimator.matrixA[:, j], bruteForceEstimator.matrixA[:, index2]))
kscore= metrics.get_k_neighbour_score(vectorb, vectorc, k)
kscores.append(kscore)
print(f"Estimating K-Score for Image {index2}: K-Score = {kscore}")

print(f"\nModel Method -- KNN-IOU score")

input_dataset = []

for i in range(len(bruteForceEstimator.imageSet)):
    img = np.array(bruteForceEstimator.imageSet[i], dtype=np.float64)
    img = torch.from_numpy(img)
    img = img.unsqueeze(0).unsqueeze(0).cuda().double()
    # img = img.repeat(1, 3, 1, 1)
    # print("size of img: ", img.shape)
    input_dataset.append(img)

stacked_tensor = torch.stack(input_dataset)
input_dataset = stacked_tensor.cpu().numpy()      
input_dataset = [torch.tensor(data).cuda().float() for data in input_dataset]
print("len(input_dataset): ",len(input_dataset))

kscores = []
vectorc=[]
vectorb = bruteForceEstimator.matrixG[index1]
for j in range(len(input_dataset)):
    input1 = model(input_dataset[j])
    input2 = model(input_dataset[index1])
    dot_product_value = torch.sum(input1 * input2, dim=1)
    vectorc.append(dot_product_value.detach().cpu().numpy().item())
kscore= metrics.get_k_neighbour_score(vectorb, vectorc, k)
kscores.append(kscore)
print(f"Estimating K-Score for Image {index1}: K-Score = {kscore}")

kscores = []
vectorc=[]
vectorb = bruteForceEstimator.matrixG[index2]
for j in range(len(input_dataset)):
    input1 = model(input_dataset[j])
    input2 = model(input_dataset[index2])
    dot_product_value = torch.sum(input1 * input2, dim=1)
    vectorc.append(dot_product_value.detach().cpu().numpy().item())
kscore= metrics.get_k_neighbour_score(vectorb, vectorc, k)
kscores.append(kscore)
print(f"Estimating K-Score for Image {index2}: K-Score = {kscore}")
    
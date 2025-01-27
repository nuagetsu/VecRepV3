''' 
No longer possible to compute matrix A given 28 dimension inputs
'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Sampler, random_split, Dataset
from torchvision import datasets, transforms
import torchvision
import torchvision.transforms as transforms

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

k=2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

class MNIST_CNN(nn.Module):
    def __init__(self, dimensions=128): 
        super(MNIST_CNN, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool1 = nn.MaxPool2d(2, 2)  
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool2 = nn.MaxPool2d(2, 2) 
        
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        
        self.relu = nn.LeakyReLU(0.1)

        self.fc1 = nn.Linear(512 * 7 * 7, 512)  
        self.fc2 = nn.Linear(512, dimensions)

    def forward(self, x):
        x = self.conv1(x)        
        x = self.bn1(x)          
        x = self.relu(x)        

        x = self.conv2(x)        
        x = self.bn2(x)          
        x = self.relu(x)        

        x = self.pool1(x)        
        
        x = self.conv3(x)        
        x = self.bn3(x)          
        x = self.relu(x)        

        x = self.conv4(x)        
        x = self.bn4(x)         
        x = self.relu(x)        

        x = self.pool2(x)       
      
        x = self.conv5(x)       
        x = self.bn5(x)        
        x = self.relu(x)       
        
        x = torch.flatten(x, start_dim=1)
  
        x = self.fc1(x)         
        x = self.relu(x)   
        x = self.fc2(x)         
        
        x = F.normalize(x, p=2, dim=1)

        return x
    
# ----------------------------------Image Input----------------------------------

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
full_dataset = torch.utils.data.ConcatDataset([trainset, testset])

# ----------------------------------Preparing the Dataset----------------------------------

def loss_fn(A, G):
    return torch.norm(A - G, p='fro')  # Frobenius norm

index1 = np.random.randint(len(full_dataset))
index2 = np.random.randint(len(full_dataset))

print("index 1: ", index1)
print("index 2: ", index2)

input1 = full_dataset[index1]
input2 = full_dataset[index2]

model = MNIST_CNN().cuda()
model.load_state_dict(torch.load('model/best_model_batch_greyscale_mnist.pth'))
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

input_dataset = []
input_images = [input1, input2] 
for i in range(len(input_images)):
    image,label = input_images[i]
    image_array = (image.numpy() > 0.5).astype(np.uint8)
    img = torch.from_numpy(image_array)
    img = img.unsqueeze(0).cuda().double()  #1x1xHxW
    # 3 channel for RGB like input
    # img = img.repeat(1, 3, 1, 1)  #1x3xHxW
    input_dataset.append(img)
    
stacked_tensor = torch.stack(input_dataset)
input_dataset = stacked_tensor.cpu().numpy()      
input_dataset = [torch.tensor(data).cuda().float() for data in input_dataset]

NCC_dataset = []
for i in range(len(input_images)):
    image,label = input_images[i]
    image_array = (image.numpy() > 0.5).astype(np.uint8)
    NCC_dataset.append(image_array)

scale = ImageProducts.scale_min(ImageProducts.ncc, -1)
NCC_scaled_value = scale(NCC_dataset[0].squeeze(0), NCC_dataset[1].squeeze(0))
print("\nCalculated values")
print("\nscaled NCC: ",NCC_scaled_value)

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

#----------------------Metric 2 - KNNIoU-----------------
print(f"\nModel Method -- KNN-IOU score")

input_dataset = []

for i in range(len(full_dataset)):
    image,label = full_dataset[i]
    image_array = (image.numpy() > 0.5).astype(np.uint8)
    img = torch.from_numpy(image_array)
    img = img.unsqueeze(0).cuda().double()  #1x1xHxW
    # 3 channel for RGB like input
    # img = img.repeat(1, 3, 1, 1)  #1x3xHxW
    input_dataset.append(img)

stacked_tensor = torch.stack(input_dataset)
input_dataset = stacked_tensor.cpu().numpy()      
input_dataset = [torch.tensor(data).cuda().float() for data in input_dataset]
# print("len(input_dataset): ",len(input_dataset))

kscores = []
vectorc=[]
vectorb = []
NCC_dataset = []

for i in range(len(full_dataset)):
    image,label = full_dataset[i]
    image_array = (image.numpy() > 0.5).astype(np.uint8)
    NCC_dataset.append(image_array)

for i in range(len(NCC_dataset)):
        scale = ImageProducts.scale_min(ImageProducts.ncc, -1)
        NCC_scaled_value = scale(NCC_dataset[index1].squeeze(0),NCC_dataset[i].squeeze(0))
        vectorb.append(NCC_scaled_value)   

print("vectorb len: ", len(vectorb))        
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
vectorb = []

for i in range(len(NCC_dataset)):
        scale = ImageProducts.scale_min(ImageProducts.ncc, -1)
        NCC_scaled_value = scale(NCC_dataset[index2].squeeze(0),NCC_dataset[i].squeeze(0))
        vectorb.append(NCC_scaled_value)   
        
for j in range(len(input_dataset)):
    input1 = model(input_dataset[j])
    input2 = model(input_dataset[index2])
    dot_product_value = torch.sum(input1 * input2, dim=1)
    vectorc.append(dot_product_value.detach().cpu().numpy().item())
kscore= metrics.get_k_neighbour_score(vectorb, vectorc, k)
kscores.append(kscore)
print(f"Estimating K-Score for Image {index2}: K-Score = {kscore}")
    
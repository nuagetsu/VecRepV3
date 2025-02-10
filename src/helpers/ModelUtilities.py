import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Sampler, random_split, Dataset
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd

import src.data_processing.ImageCalculations as imgcalc
import src.visualization.Metrics as metrics
import src.visualization.ImagePlots as imgplt
import src.data_processing.ImageProducts as ImageProducts

from functools import partial
from learnable_polyphase_sampling.learn_poly_sampling.layers import get_logits_model, PolyphaseInvariantDown2D, LPS
from learnable_polyphase_sampling.learn_poly_sampling.layers.polydown import set_pool

def loss_fn_frobenius(A, G):
    return torch.norm(A - G, p='fro')  

def loss_fn_MSE(A,G):
    return F.mse_loss(A, G)

# ------------------- For dimension of 128 ----------------------
# class SimpleCNN(nn.Module):
#     def __init__(self, dimensions=128, padding_mode='zeros'):
#         super().__init__()
#         self.conv1 = nn.Conv2d(1, 32, 3, padding=1, padding_mode=padding_mode)
#         self.lpd = set_pool(partial(
#             PolyphaseInvariantDown2D,
#             component_selection=LPS,
#             get_logits=get_logits_model('LPSLogitLayers'),
#             pass_extras=False
#             ),p_ch=256,h_ch=256)

#         self.bn1   = nn.BatchNorm2d(32)
#         self.relu = nn.LeakyReLU(0.1)
        
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, padding_mode=padding_mode)
#         self.bn2   = nn.BatchNorm2d(64)
        
#         self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, padding_mode=padding_mode)
#         self.bn3   = nn.BatchNorm2d(128)
        
#         self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1, padding_mode=padding_mode)
#         self.bn4   = nn.BatchNorm2d(256)
        
#         self.avgpool=nn.AdaptiveAvgPool2d((1,1))
#         self.fc=nn.Linear(256, dimensions)
        
#     def forward(self,x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
        
#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = self.relu(x)
        
#         x = self.conv3(x)
#         x = self.bn3(x)
#         x = self.relu(x)
        
#         x = self.conv4(x)
#         x = self.bn4(x)
#         x = self.relu(x)
        
#         x = self.lpd(x)  # Use just as any down-sampling layer!
#         x = torch.flatten(self.avgpool(x),1)
#         x = self.fc(x)
#         x = F.normalize(x, p=2, dim=1)
#         return x


class SimpleCNN(nn.Module):
    def __init__(self, dimensions=32, padding_mode='circular'):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1, padding_mode=padding_mode)
        self.lpd = set_pool(partial(
            PolyphaseInvariantDown2D,
            component_selection=LPS,
            get_logits=get_logits_model('LPSLogitLayers'),
            pass_extras=False
            ),p_ch=128,h_ch=128)

        self.bn1   = nn.BatchNorm2d(32)
        self.relu = nn.LeakyReLU(0.1)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, padding_mode=padding_mode)
        self.bn2   = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, padding_mode=padding_mode)
        self.bn3   = nn.BatchNorm2d(128)
        
        
        self.avgpool=nn.AdaptiveAvgPool2d((1,1))
        self.fc=nn.Linear(128, dimensions)
        
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        
        x = self.lpd(x)  # Use just as any down-sampling layer!
        x = torch.flatten(self.avgpool(x),1)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        return x

class CNN(nn.Module):
    def __init__(self, dimensions=128, padding_mode='circular'):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1, padding_mode=padding_mode)
        self.lpd = set_pool(partial(
            PolyphaseInvariantDown2D,
            component_selection=LPS,
            get_logits=get_logits_model('LPSLogitLayers'),
            pass_extras=False
            ),p_ch=512,h_ch=512)

        self.bn1   = nn.BatchNorm2d(32)
        self.relu = nn.LeakyReLU(0.1)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, padding_mode=padding_mode)
        self.bn2   = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, padding_mode=padding_mode)
        self.bn3   = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1, padding_mode=padding_mode)
        self.bn4   = nn.BatchNorm2d(256)
        
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1, padding_mode=padding_mode) 
        self.bn5 = nn.BatchNorm2d(512)
        
        self.avgpool=nn.AdaptiveAvgPool2d((1,1))
        self.fc=nn.Linear(512, dimensions)
        
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)
        
        x = self.lpd(x)  # Use just as any down-sampling layer!
        x = torch.flatten(self.avgpool(x),1)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        return x
    
    
class ComplexCNN(nn.Module):
    def __init__(self, dimensions=128, padding_mode='circular'):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1, padding_mode=padding_mode)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, padding_mode=padding_mode)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, padding_mode=padding_mode)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1, padding_mode=padding_mode)
        self.bn4 = nn.BatchNorm2d(256)

        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1, padding_mode=padding_mode) 
        self.bn5 = nn.BatchNorm2d(512)

        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, padding=1, padding_mode=padding_mode)  
        self.bn6 = nn.BatchNorm2d(512)

        self.lpd = set_pool(partial(
            PolyphaseInvariantDown2D,
            component_selection=LPS,
            get_logits=get_logits_model('LPSLogitLayers'),
            pass_extras=False
        ), p_ch=512, h_ch=512) 

        self.relu = nn.LeakyReLU(0.1)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, dimensions)  
        
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x))) 
        x = self.relu(self.bn6(self.conv6(x))) 
        
        x = self.lpd(x)  
        x = torch.flatten(self.avgpool(x), 1)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        return x
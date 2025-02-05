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

import src.data_processing.BruteForceEstimator as bfEstimator
import src.data_processing.ImageCalculations as imgcalc
import src.visualization.BFmethod as graphing
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

#original
class MNIST_CNN(nn.Module):
    def __init__(self, dimensions=32):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),  # [B, 32, 28, 28]
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),  # [B, 32, 14, 14]
            
            nn.Conv2d(32, 64, 3, padding=1),  # [B, 64, 14, 14]
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),  # [B, 64, 7, 7]
            
            #should i do this
            nn.AdaptiveAvgPool2d(1)  # [B, 64, 1, 1]
        )
        self.fc = nn.Linear(64, dimensions)
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)  # [B, 64]
        x = self.fc(x)  # [B, dimensions]
        return F.normalize(x, p=2, dim=1)
    

class SimpleCNN(nn.Module):
    def __init__(self, dimensions=32):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),  # [B, 32, 28, 28]
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),  # [B, 32, 14, 14]
            
            nn.Conv2d(32, 64, 3, padding=1),  # [B, 64, 14, 14]
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),  # [B, 64, 7, 7]
            
            #should i do this
            nn.AdaptiveAvgPool2d(1)  # [B, 64, 1, 1]
        )
        self.fc = nn.Linear(64, dimensions)
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)  # [B, 64]
        x = self.fc(x)  # [B, dimensions]
        return F.normalize(x, p=2, dim=1)
    
class CNN(nn.Module):
    def __init__(self, dimensions=128):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)

        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)
        
        self.lpd = set_pool(partial(
            PolyphaseInvariantDown2D,
            component_selection=LPS,
            get_logits=get_logits_model('LPSLogitLayers'),
            pass_extras=False
            ),p_ch=128,h_ch=128)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4   = nn.BatchNorm2d(256)
        
        self.relu = nn.LeakyReLU(0.1)
        
#         self.adapt_pool = nn.AdaptiveAvgPool2d((4, 4)) 

#         self.fc1 = nn.Linear(256*4*4, dimensions)
        
        #new model training for later
        self.adapt_pool = nn.AdaptiveAvgPool2d((1, 1)) 

        self.fc1 = nn.Linear(256, dimensions)
    
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
        
        x = self.lpd(x)    # 16x16 -> 8x8 
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x) 
        
        x = self.adapt_pool(x) 
        
        x = torch.flatten(x, start_dim=1)  
        x = self.fc1(x)                    
        
        x = F.normalize(x, p=2, dim=1)
        return x
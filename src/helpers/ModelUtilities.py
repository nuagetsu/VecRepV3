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
import src.helpers.ModelUtilities as models

def loss_fn_frobenius(A, G):
    return torch.norm(A - G, p='fro')  

def loss_fn_MSE(A,G):
    return F.mse_loss(A, G)

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
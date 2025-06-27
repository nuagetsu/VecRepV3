import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader, TensorDataset, Sampler, random_split, Dataset
from torchvision import datasets, transforms


import matplotlib.pyplot as plt
import numpy as np
import math
import cmath
import pandas as pd

import src.data_processing.ImageCalculations as imgcalc
import src.visualization.Metrics as metrics
import src.visualization.ImagePlots as imgplt
import src.data_processing.ImageProducts as ImageProducts

from scipy.integrate import dblquad

from functools import partial
from learnable_polyphase_sampling.learn_poly_sampling.layers import get_logits_model, PolyphaseInvariantDown2D, LPS
from learnable_polyphase_sampling.learn_poly_sampling.layers.polydown import set_pool

from truly_shift_invariant_cnns.models.aps_models.apspool import ApsPool 

def loss_fn_frobenius(A, G):
    return torch.norm(A - G, p='fro')  

def loss_fn_MSE(A,G):
    return F.mse_loss(A, G)



class CirclePruningMethod(prune.BasePruningMethod):
    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold
    """Prune every other entry in a tensor
    """
    PRUNING_TYPE = 'structured'

    def metric_s1(self, x, y):
        return cmath.acos(np.dot(x, np.conj(y))).real

    def compute_mask(self, t, default_mask):
        mask = default_mask.clone()
        out_channels = mask.shape[0]
        in_channels = mask.shape[1]
        kernel_h = mask.shape[2]
        kernel_w = mask.shape[3]
        for i in range(out_channels):
            for k in range(in_channels):
                x = cmath.exp((2 * cmath.pi * 1j * i) / in_channels)
                y = cmath.exp((2 * cmath.pi * 1j * k) / out_channels)
                distance = self.metric_s1(x, y)
                if (distance > self.threshold):
                    for n in range(kernel_h):
                        for m in range(kernel_w):
                            mask[i][k][n][m] = 0

        #print(mask)
        return mask

class KleinPruningMethod(prune.BasePruningMethod):
    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold
    """Prune every other entry in a tensor
    """
    PRUNING_TYPE = 'structured'

    def metric_k(self, t1, t2, m1, m2):
        Q = lambda t: (2 * math.pow(t, 2)) - 1
        F_embed = lambda t1, t2: lambda x, y: (math.sin(t2) * (math.cos(t1) * x + math.sin(t1) * y)) + (math.cos(t2) * Q((math.cos(t1) * x) + math.sin(t1) * y))
        func = lambda t1, t2, m1, m2: lambda x, y: (F_embed(t1, t2)(x, y) - F_embed(m1, m2)(x, y)) ** 2
        return math.sqrt(
            dblquad(func(t1, t2, m1, m2), -1, 1, -1, 1)[0]
        )

    def compute_mask(self, t, default_mask):
        mask = default_mask.clone()
        out_channels = mask.shape[0]
        in_channels = mask.shape[1]
        d1_size = math.floor(math.sqrt(in_channels)) # discretized grid size for in klein bottle. Note this only supports square nums for now...
        d2_size = math.floor(math.sqrt(out_channels))
        kernel_h = mask.shape[2]
        kernel_w = mask.shape[3]
        # print(mask.shape)
        # print((d1_size, d2_size))
        for i in range(d1_size):
            for k in range(d1_size):
                for a in range(d2_size):
                    for b in range(d2_size):
                        t1 = math.pi * (i / d1_size)
                        t2 = math.pi * (k / d1_size)
                        m1 = math.pi * (a / d2_size)
                        m2 = math.pi * (b / d2_size)
                        distance = self.metric_k(t1, t2, m1, m2)
                        if (distance > self.threshold):
                            for n in range(kernel_h):
                                for m in range(kernel_w):
                                    mask[(a * d2_size) + b][(i * d1_size) + k][n][m] = 0

        #print(mask)
        return mask

def circle_filter_generator(in_channels, out_channels, kernel_size):
    # in_channel is implicitly 1
    size = (2 * kernel_size) + 1
    F = lambda x, y: (math.cos(theta) * x) + (math.sin(theta) * y)
    filter = np.ndarray((out_channels, in_channels, size, size))
    for i in range(out_channels):
        for j in range(in_channels):
            theta = 2 * math.pi * (i / out_channels)
            for n in range(size):
                for m in range(size):
                    lb_y = -1 + ((2 * m) / size)
                    ub_y = -1 + ((2 * (m + 1)) / size)
                    lb_x = lambda y: (-1 + ((2 * n) / size))
                    ub_x = lambda y: (-1 + ((2 * (n+1)) / size))
                    filter[i][j][n][m] = dblquad(F, lb_y, ub_y, lb_x, ub_x)[0]
    
    return filter

def klein_filter_generator(in_channels, out_channels, kernel_size):
    # in_channels implicitly 1
    # ok it's working now!
    size = (2 * kernel_size) + 1
    Q = lambda t: (2 * math.pow(t, 2)) - 1
    F_embed = lambda t1, t2: lambda x, y: (math.sin(t2) * (math.cos(t1) * x + math.sin(t1) * y)) + (math.cos(t2) * Q((math.cos(t1) * x) + math.sin(t1) * y))
    filter = np.ndarray((out_channels, in_channels, size, size))
    d_size = math.floor(math.sqrt(out_channels)) # size of discretised grid on klein bottle
    for i in range(d_size):
        for k in range(d_size):
            for j in range(in_channels):
                theta1 = math.pi * (i / d_size)
                theta2 = math.pi * (k / d_size)
                for n in range(size):
                    for m in range(size):
                        lb_y = -1 + ((2 * m) / size)
                        ub_y = -1 + ((2 * (m + 1)) / size)
                        lb_x = lambda y: (-1 + ((2 * n) / size))
                        ub_x = lambda y: (-1 + ((2 * (n+1)) / size))
                        filter[i * (d_size) + k][j][n][m] = dblquad(F_embed(theta1, theta2), lb_y, ub_y, lb_x, ub_x)[0]
    
    return filter

class KleinFilters(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, dimensions=10, padding_mode='circular'):
        super().__init__()
        # Klein filter layer with 1 input channel, 16 output channels, kernel size 3. This represents a discretised klein bottle w theta1 and theta2 split into 4.
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size*2 + 1), padding=padding, padding_mode=padding_mode) 
        with torch.no_grad():
             self.conv1.weight = nn.Parameter(torch.from_numpy(klein_filter_generator(in_channels, out_channels, kernel_size)).float())
        
        self.lpd1 = set_pool(partial(
            PolyphaseInvariantDown2D,
            component_selection=LPS,
            get_logits=get_logits_model('LPSLogitLayers'),
            pass_extras=False
            ),p_ch=out_channels,h_ch=out_channels)

        self.bn1   = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.1)
        

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.lpd1(x)  # Use just as any down-sampling layer!
        return x
    




class CircleFilters(nn.Module):
    def __init__(self, in_channels, out_channels, dimensions=10, padding_mode='circular'):
        super().__init__()
        #print(in_channels)
        # Circle filter layer with 1 input channel, 16 output channels, kernel size 3. This represents a discretised circle with roots of unity 16
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=1, padding_mode=padding_mode) 
        #print(self.cf1.groups)
        with torch.no_grad():
            self.conv1.weight = nn.Parameter(torch.from_numpy(circle_filter_generator(in_channels, out_channels, 3)).float())
        
        self.lpd1 = set_pool(partial(
            PolyphaseInvariantDown2D,
            component_selection=LPS,
            get_logits=get_logits_model('LPSLogitLayers'),
            pass_extras=False
            ),p_ch=out_channels,h_ch=out_channels)

        self.bn1   = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.1)


    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.lpd1(x)  # Use just as any down-sampling layer!
        return x

class CircleOneLayer(nn.Module):
    def __init__(self, threshold, in_channels, out_channels, dimensions=10, padding_mode='circular'):
        super().__init__()
        self.threshold = threshold
        
        self.c1 = CircleFilters(in_channels, out_channels)
        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, padding_mode=padding_mode)
        self.prune = CirclePruningMethod(threshold)
        self.prune.apply(self.conv1, name='weight', threshold=self.threshold)
        self.lpd1 = set_pool(partial(
            PolyphaseInvariantDown2D,
            component_selection=LPS,
            get_logits=get_logits_model('LPSLogitLayers'),
            pass_extras=False
            ),p_ch=out_channels,h_ch=out_channels)

        self.bn1   = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.1)
        
    def forward(self,x):
        x = self.c1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.lpd1(x)  # Use just as any down-sampling layer!
        return x
    
class KleinOneLayer(nn.Module):
    def __init__(self,threshold, in_channels, out_channels, kernel_size, padding, dimensions=10, padding_mode='circular'):
        super().__init__()
        self.threshold = threshold
        
        self.k1 = KleinFilters(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=padding, padding_mode=padding_mode)
        self.prune = KleinPruningMethod(threshold)
        self.prune.apply(self.conv1, name='weight', threshold=self.threshold)
        self.lpd1 = set_pool(partial(
            PolyphaseInvariantDown2D,
            component_selection=LPS,
            get_logits=get_logits_model('LPSLogitLayers'),
            pass_extras=False
            ),p_ch=out_channels,h_ch=out_channels)

        self.bn1   = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.1)
        
    def forward(self,x):
        x = self.k1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.lpd1(x)  # Use just as any down-sampling layer!
        return x
        

class KleinLayers2(nn.Module):
    def __init__(self, threshold, dimensions=10, padding_mode='circular'):
        super().__init__()
        self.kol1 = KleinOneLayer(threshold, 1, 16)
        self.kol2 = KleinOneLayer(threshold, 16, 32)
        
        self.avgpool=nn.AdaptiveAvgPool2d((1,1))
        self.fc=nn.Linear(32, dimensions)
        
    def forward(self, x):
        x = self.kol1(x)
        x = self.kol2(x)

        x = torch.flatten(self.avgpool(x),1)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        return x
    
class CircleLayers2(nn.Module):
    def __init__(self, threshold, dimensions=10, padding_mode='circular'):
        super().__init__()
        self.col1 = CircleOneLayer(threshold, 1, 16)
        self.col2 = CircleOneLayer(threshold, 16, 32)
        
        self.avgpool=nn.AdaptiveAvgPool2d((1,1))
        self.fc=nn.Linear(32, dimensions)
        
    def forward(self, x):
        x = self.col1(x)
        x = self.col2(x)

        x = torch.flatten(self.avgpool(x),1)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        return x



class SimpleCNN1(nn.Module):
    def __init__(self, dimensions=10, padding_mode='circular'):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1, padding_mode=padding_mode)
        self.lpd1 = set_pool(partial(
            PolyphaseInvariantDown2D,
            component_selection=LPS,
            get_logits=get_logits_model('LPSLogitLayers'),
            pass_extras=False
            ),p_ch=16,h_ch=16)

        self.bn1   = nn.BatchNorm2d(16)
        self.relu = nn.LeakyReLU(0.1)
             
        self.avgpool=nn.AdaptiveAvgPool2d((1,1))
        self.fc=nn.Linear(16, dimensions)
        
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.lpd1(x)  # Use just as any down-sampling layer!
        x = torch.flatten(self.avgpool(x),1)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        return x
    

class SimpleCNN4_2fc(nn.Module):
    def __init__(self, dimensions=10, padding_mode='circular'):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1, padding_mode=padding_mode)
        self.lpd1 = set_pool(partial(
                    PolyphaseInvariantDown2D,
                    component_selection=LPS,
                    get_logits=get_logits_model('LPSLogitLayers'),
                    pass_extras=False
                    ),p_ch=16,h_ch=16)
        self.bn1   = nn.BatchNorm2d(16)
        self.relu = nn.LeakyReLU(0.1)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1, padding_mode=padding_mode)
        self.lpd2 = set_pool(partial(
            PolyphaseInvariantDown2D,
            component_selection=LPS,
            get_logits=get_logits_model('LPSLogitLayers'),
            pass_extras=False
            ),p_ch=32,h_ch=32)        
        self.bn2   = nn.BatchNorm2d(32)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1, padding_mode=padding_mode)
        self.lpd3 = set_pool(partial(
            PolyphaseInvariantDown2D,
            component_selection=LPS,
            get_logits=get_logits_model('LPSLogitLayers'),
            pass_extras=False
            ),p_ch=64,h_ch=64)        
        self.bn3   = nn.BatchNorm2d(64)
        
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1, padding_mode=padding_mode)
        self.lpd4 = set_pool(partial(
            PolyphaseInvariantDown2D,
            component_selection=LPS,
            get_logits=get_logits_model('LPSLogitLayers'),
            pass_extras=False
            ),p_ch=128,h_ch=128)
        self.bn4   = nn.BatchNorm2d(128)
        
        self.avgpool=nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(128, 128)
        self.ln1 = nn.LayerNorm(128)
        self.fc2 = nn.Linear(128, dimensions)
        
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.lpd1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.lpd2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.lpd3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.lpd4(x)
        
        x = torch.flatten(self.avgpool(x),1)
        x = self.fc1(x)
        x = self.ln1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = F.normalize(x, p=2, dim=1)
        return x 

class Klein2_2fc(nn.Module):
    def __init__(self, threshold, dimensions=10, padding_mode='circular'):
        super().__init__()
        self.kol1 = KleinOneLayer(threshold, 1, 16, 3)
        self.kol2 = KleinOneLayer(threshold, 16, 32, 3)
        self.relu = nn.LeakyReLU(0.1)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(32, dimensions)
        self.ln1 = nn.LayerNorm(128)
        self.fc2 = nn.Linear(32, dimensions)
        
    def forward(self, x):
        x = self.kol1(x)
        x = self.kol2(x)

        x = torch.flatten(self.avgpool(x),1)
        x = self.fc1(x)
        x = self.ln1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = F.normalize(x, p=2, dim=1)
        return x 

class Klein4_2fc(nn.Module):
    def __init__(self, threshold, dimensions=10, padding_mode='circular'):
        super().__init__()
        self.kol1 = KleinOneLayer(threshold, 1, 16, 3, 1)
        self.kol2 = KleinOneLayer(threshold, 16, 32, 3, 1)
        self.kol3 = KleinOneLayer(threshold, 32, 64, 1, 1)
        self.kol4 = KleinOneLayer(threshold, 64, 128, 1, 1)
        self.relu = nn.LeakyReLU(0.1)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(128, dimensions)
        self.ln1 = nn.LayerNorm(128)
        self.fc2 = nn.Linear(128, dimensions)
        
    def forward(self, x):
        x = self.kol1(x)
        x = self.kol2(x)
        x = self.kol3(x)
        x = self.kol4(x)

        x = torch.flatten(self.avgpool(x),1)
        x = self.fc1(x)
        x = self.ln1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = F.normalize(x, p=2, dim=1)
        return x 
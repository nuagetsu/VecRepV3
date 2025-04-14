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

from truly_shift_invariant_cnns.models.aps_models.apspool import ApsPool 

def loss_fn_frobenius(A, G):
    return torch.norm(A - G, p='fro')  

def loss_fn_MSE(A,G):
    return F.mse_loss(A, G)

##################################################################33
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
    
class SimpleCNN2(nn.Module):
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
        
        self.avgpool=nn.AdaptiveAvgPool2d((1,1))
        self.fc=nn.Linear(32, dimensions)
        
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.lpd1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.lpd2(x)  # Use just as any down-sampling layer!
        x = torch.flatten(self.avgpool(x),1)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        return x
    
class SimpleCNN3(nn.Module):
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

        self.avgpool=nn.AdaptiveAvgPool2d((1,1))
        self.fc=nn.Linear(64, dimensions)
        
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
        x = self.lpd3(x)  # Use just as any down-sampling layer!
        x = torch.flatten(self.avgpool(x),1)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        return x

class SimpleCNN4(nn.Module):
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
        self.fc=nn.Linear(128, dimensions)
        
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
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        return x    
    
class SimpleCNN6(nn.Module):
    def __init__(self, dimensions=128, padding_mode='circular'):
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

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1, padding_mode=padding_mode) 
        self.lpd5 = set_pool(partial(
            PolyphaseInvariantDown2D,
            component_selection=LPS,
            get_logits=get_logits_model('LPSLogitLayers'),
            pass_extras=False
            ),p_ch=256,h_ch=256)
        self.bn5 = nn.BatchNorm2d(256)

        self.conv6 = nn.Conv2d(256, 512, kernel_size=3, padding=1, padding_mode=padding_mode)  
        self.lpd6 = set_pool(partial(
            PolyphaseInvariantDown2D,
            component_selection=LPS,
            get_logits=get_logits_model('LPSLogitLayers'),
            pass_extras=False
            ),p_ch=512,h_ch=512)
        self.bn6 = nn.BatchNorm2d(512)

        self.relu = nn.LeakyReLU(0.1)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, dimensions)  
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        #x = self.lpd1(x)
        
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
        
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)
        x = self.lpd5(x)
        
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu(x)
        x = self.lpd6(x)
        
        x = torch.flatten(self.avgpool(x),1)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        return x
    
class SimpleCNN8(nn.Module):
    def __init__(self, dimensions=128, padding_mode='circular'):
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

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1, padding_mode=padding_mode) 
        self.lpd5 = set_pool(partial(
            PolyphaseInvariantDown2D,
            component_selection=LPS,
            get_logits=get_logits_model('LPSLogitLayers'),
            pass_extras=False
            ),p_ch=256,h_ch=256)
        self.bn5 = nn.BatchNorm2d(256)

        self.conv6 = nn.Conv2d(256, 512, kernel_size=3, padding=1, padding_mode=padding_mode)  
        self.lpd6 = set_pool(partial(
            PolyphaseInvariantDown2D,
            component_selection=LPS,
            get_logits=get_logits_model('LPSLogitLayers'),
            pass_extras=False
            ),p_ch=512,h_ch=512)
        self.bn6 = nn.BatchNorm2d(512)
                
        self.conv7 = nn.Conv2d(512, 1028, kernel_size=3, padding=1, padding_mode=padding_mode)  
        self.lpd7 = set_pool(partial(
            PolyphaseInvariantDown2D,
            component_selection=LPS,
            get_logits=get_logits_model('LPSLogitLayers'),
            pass_extras=False
            ),p_ch=1028,h_ch=1028)
        self.bn7 = nn.BatchNorm2d(1028)
                
        self.conv8 = nn.Conv2d(1028, 2056, kernel_size=3, padding=1, padding_mode=padding_mode)  
        self.lpd8 = set_pool(partial(
            PolyphaseInvariantDown2D,
            component_selection=LPS,
            get_logits=get_logits_model('LPSLogitLayers'),
            pass_extras=False
            ),p_ch=2056,h_ch=2056)
        self.bn8 = nn.BatchNorm2d(2056)

        self.relu = nn.LeakyReLU(0.1)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(2056, dimensions)  
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        #x = self.lpd1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        #x = self.lpd2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        #x = self.lpd3(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.lpd4(x)
        
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)
        x = self.lpd5(x)
        
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu(x)
        x = self.lpd6(x)
        
        x = self.conv7(x)
        x = self.bn7(x)
        x = self.relu(x)
        x = self.lpd7(x)
        
        x = self.conv8(x)
        x = self.bn8(x)
        x = self.relu(x)
        x = self.lpd8(x)
        x = torch.flatten(self.avgpool(x),1)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        return x 

class SimpleCNN2_aps(nn.Module):
    def __init__(self, dimensions=10, padding_mode='circular'):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1, padding_mode=padding_mode)
        self.apspool1 = ApsPool(16, return_poly_indices=False)
        self.bn1   = nn.BatchNorm2d(16)
        self.relu = nn.LeakyReLU(0.1)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1, padding_mode=padding_mode)
        self.apspool2 = ApsPool(32, return_poly_indices=False)
        self.bn2   = nn.BatchNorm2d(32)
        
        self.avgpool=nn.AdaptiveAvgPool2d((1,1))
        self.fc=nn.Linear(32, dimensions)
        
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.apspool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.apspool2(x)  # Use just as any down-sampling layer!
        x = torch.flatten(self.avgpool(x),1)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        return x
    
class SimpleCNN4_aps(nn.Module):
    def __init__(self, dimensions=10, padding_mode='circular'):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1, padding_mode=padding_mode)
        self.apspool1 = ApsPool(16, return_poly_indices=False)
        self.bn1   = nn.BatchNorm2d(16)
        self.relu = nn.LeakyReLU(0.1)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1, padding_mode=padding_mode)
        self.apspool2 = ApsPool(32, return_poly_indices=False)     
        self.bn2   = nn.BatchNorm2d(32)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1, padding_mode=padding_mode)
        self.apspool3 = ApsPool(64, return_poly_indices=False)      
        self.bn3   = nn.BatchNorm2d(64)
        
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1, padding_mode=padding_mode)
        self.apspool4 = ApsPool(128, return_poly_indices=False)
        self.bn4   = nn.BatchNorm2d(128)
        
        self.avgpool=nn.AdaptiveAvgPool2d((1,1))
        self.fc=nn.Linear(128, dimensions)
        
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.apspool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.apspool2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.apspool3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.apspool4(x)
        
        x = torch.flatten(self.avgpool(x),1)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        return x    
    
class SimpleCNN6_aps(nn.Module):
    def __init__(self, dimensions=128, padding_mode='circular'):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 16, 3, padding=1, padding_mode=padding_mode)
        self.apspool1 = ApsPool(16, return_poly_indices=False)
        self.bn1   = nn.BatchNorm2d(16)
        self.relu = nn.LeakyReLU(0.1)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1, padding_mode=padding_mode)
        self.apspool2 = ApsPool(32, return_poly_indices=False)     
        self.bn2   = nn.BatchNorm2d(32)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1, padding_mode=padding_mode)
        self.apspool3 = ApsPool(64, return_poly_indices=False)      
        self.bn3   = nn.BatchNorm2d(64)
        
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1, padding_mode=padding_mode)
        self.apspool4 = ApsPool(128, return_poly_indices=False)
        self.bn4   = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1, padding_mode=padding_mode) 
        self.apspool5 = ApsPool(256, return_poly_indices=False)
        self.bn5 = nn.BatchNorm2d(256)

        self.conv6 = nn.Conv2d(256, 512, kernel_size=3, padding=1, padding_mode=padding_mode)  
        self.apspool6 = ApsPool(512, return_poly_indices=False)
        self.bn6 = nn.BatchNorm2d(512)

        self.relu = nn.LeakyReLU(0.1)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, dimensions)  
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        #x = self.apspool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.apspool2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.apspool3(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.apspool4(x)
        
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)
        x = self.apspool5(x)
        
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu(x)
        x = self.apspool6(x)
        
        x = torch.flatten(self.avgpool(x),1)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        return x
    
######################################################################################
#CBAM implementation
class CAM(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction_ratio, channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x).squeeze(-1).squeeze(-1))
        max_out = self.mlp(self.max_pool(x).squeeze(-1).squeeze(-1))
        channel_attention = self.sigmoid(avg_out + max_out)
        return x * channel_attention.unsqueeze(-1).unsqueeze(-1)

class SAM(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_attention = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * spatial_attention

class CBAM(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super().__init__()
        self.channel_att = CAM(channels, reduction_ratio)
        self.spatial_att = SAM()

    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x

class SimpleCNN2_aps_CBAM(nn.Module):
    def __init__(self, dimensions=10, padding_mode='circular'):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1, padding_mode=padding_mode)
        self.bn1 = nn.BatchNorm2d(16)
        self.cbam1 = CBAM(16, reduction_ratio=4)  # Channel attention with reduction ratio 4
        self.apspool1 = ApsPool(16, return_poly_indices=False)
        self.relu = nn.LeakyReLU(0.1)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1, padding_mode=padding_mode)
        self.bn2 = nn.BatchNorm2d(32)
        self.cbam2 = CBAM(32, reduction_ratio=8)  # Channel attention with reduction ratio 8
        self.apspool2 = ApsPool(32, return_poly_indices=False)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(32, dimensions)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.cbam1(x)
        x = self.relu(x)
        x = self.apspool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.cbam2(x)  
        x = self.relu(x)
        x = self.apspool2(x)
        
        x = torch.flatten(self.avgpool(x), 1)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        return x

class SimpleCNN2_aps_CBAM_dropout(nn.Module):
    def __init__(self, dimensions=10, padding_mode='circular', dropout_rate=0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1, padding_mode=padding_mode)
        self.bn1 = nn.BatchNorm2d(16)
        self.cbam1 = CBAM(16, reduction_ratio=4)
        self.apspool1 = ApsPool(16, return_poly_indices=False)
        self.relu = nn.LeakyReLU(0.1)  
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1, padding_mode=padding_mode)
        self.bn2 = nn.BatchNorm2d(32)
        self.cbam2 = CBAM(32, reduction_ratio=8)
        self.apspool2 = ApsPool(32, return_poly_indices=False)
        
        self.dropout = nn.Dropout2d(p=dropout_rate) 
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, dimensions)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.cbam1(x)
        x = self.relu(x)
        x = self.apspool1(x)
        x = self.dropout(x) 
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.cbam2(x)
        x = self.relu(x)
        x = self.apspool2(x)
        x = self.dropout(x) 
        
        x = torch.flatten(self.avgpool(x), 1)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        
        return x

class SimpleCNN4_aps_dropout(nn.Module):
    def __init__(self, dimensions=10, padding_mode='circular', dropout_rate=0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1, padding_mode=padding_mode)
        self.bn1 = nn.BatchNorm2d(16)
        self.apspool1 = ApsPool(16, return_poly_indices=False)
        self.relu = nn.LeakyReLU(0.1)  
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1, padding_mode=padding_mode)
        self.bn2 = nn.BatchNorm2d(32)
        self.apspool2 = ApsPool(32, return_poly_indices=False)     
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1, padding_mode=padding_mode)
        self.bn3 = nn.BatchNorm2d(64)
        self.apspool3 = ApsPool(64, return_poly_indices=False)     
        
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1, padding_mode=padding_mode)
        self.bn4 = nn.BatchNorm2d(128)
        self.apspool4 = ApsPool(128, return_poly_indices=False)     
        
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, dimensions)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.apspool1(x)
        x = self.dropout(x) 
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.apspool2(x)
        x = self.dropout(x) 
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.apspool3(x)
        x = self.dropout(x) 
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.apspool4(x)
        x = self.dropout(x) 
        
        x = torch.flatten(self.avgpool(x), 1)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        
        return x
    
class SimpleCNN4_aps_CBAM_dropout(nn.Module):
    def __init__(self, dimensions=10, padding_mode='circular', dropout_rate=0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1, padding_mode=padding_mode)
        self.bn1 = nn.BatchNorm2d(16)
        self.cbam1 = CBAM(16, reduction_ratio=4)
        self.apspool1 = ApsPool(16, return_poly_indices=False)
        self.relu = nn.LeakyReLU(0.1)  
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1, padding_mode=padding_mode)
        self.bn2 = nn.BatchNorm2d(32)
        self.cbam2 = CBAM(32, reduction_ratio=8)
        self.apspool2 = ApsPool(32, return_poly_indices=False)     
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1, padding_mode=padding_mode)
        self.bn3 = nn.BatchNorm2d(64)
        self.cbam3 = CBAM(64, reduction_ratio=16)
        self.apspool3 = ApsPool(64, return_poly_indices=False)     
        
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1, padding_mode=padding_mode)
        self.bn4 = nn.BatchNorm2d(128)
        self.cbam4 = CBAM(128, reduction_ratio=32)
        self.apspool4 = ApsPool(128, return_poly_indices=False)     
        
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, dimensions)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.cbam1(x)
        x = self.relu(x)
        x = self.apspool1(x)
        x = self.dropout(x) 
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.cbam2(x)
        x = self.relu(x)
        x = self.apspool2(x)
        x = self.dropout(x) 
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.cbam3(x)
        x = self.relu(x)
        x = self.apspool3(x)
        x = self.dropout(x) 
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.cbam4(x)
        x = self.relu(x)
        x = self.apspool4(x)
        x = self.dropout(x) 
        
        x = torch.flatten(self.avgpool(x), 1)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        
        return x

class SimpleCNN6_aps_CBAM_dropout(nn.Module):
    def __init__(self, dimensions=10, padding_mode='circular', dropout_rate=0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1, padding_mode=padding_mode)
        self.bn1 = nn.BatchNorm2d(16)
        self.cbam1 = CBAM(16, reduction_ratio=4)
        self.apspool1 = ApsPool(16, return_poly_indices=False)
        self.relu = nn.LeakyReLU(0.1)  
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1, padding_mode=padding_mode)
        self.bn2 = nn.BatchNorm2d(32)
        self.cbam2 = CBAM(32, reduction_ratio=8)
        self.apspool2 = ApsPool(32, return_poly_indices=False)     
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1, padding_mode=padding_mode)
        self.bn3 = nn.BatchNorm2d(64)
        self.cbam3 = CBAM(64, reduction_ratio=16)
        self.apspool3 = ApsPool(64, return_poly_indices=False)     
        
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1, padding_mode=padding_mode)
        self.bn4 = nn.BatchNorm2d(128)
        self.cbam4 = CBAM(128, reduction_ratio=32)
        self.apspool4 = ApsPool(128, return_poly_indices=False)   
        
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1, padding_mode=padding_mode) 
        self.apspool5 = ApsPool(256, return_poly_indices=False)
        self.bn5 = nn.BatchNorm2d(256)
        self.cbam5 = CBAM(256, reduction_ratio=64)

        self.conv6 = nn.Conv2d(256, 512, kernel_size=3, padding=1, padding_mode=padding_mode)  
        self.apspool6 = ApsPool(512, return_poly_indices=False)
        self.bn6 = nn.BatchNorm2d(512)
        self.cbam6 = CBAM(512, reduction_ratio=128)
        
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, dimensions)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.cbam1(x)
        x = self.relu(x)
        #x = self.apspool1(x)
        x = self.dropout(x) 
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.cbam2(x)
        x = self.relu(x)
        x = self.apspool2(x)
        x = self.dropout(x) 
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.cbam3(x)
        x = self.relu(x)
        x = self.apspool3(x)
        x = self.dropout(x) 
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.cbam4(x)
        x = self.relu(x)
        x = self.apspool4(x)
        x = self.dropout(x) 
        
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.cbam5(x)
        x = self.relu(x)
        x = self.apspool5(x)
        x = self.dropout(x) 
        
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.cbam6(x)
        x = self.relu(x)
        x = self.apspool6(x)
        x = self.dropout(x)
        
        x = torch.flatten(self.avgpool(x), 1)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)    
        
        return x
    
class SimpleCNN6_aps_dropout(nn.Module):
    def __init__(self, dimensions=10, padding_mode='circular', dropout_rate=0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1, padding_mode=padding_mode)
        self.bn1 = nn.BatchNorm2d(16)
        self.apspool1 = ApsPool(16, return_poly_indices=False)
        self.relu = nn.LeakyReLU(0.1)  
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1, padding_mode=padding_mode)
        self.bn2 = nn.BatchNorm2d(32)
        self.apspool2 = ApsPool(32, return_poly_indices=False)     
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1, padding_mode=padding_mode)
        self.bn3 = nn.BatchNorm2d(64)
        self.apspool3 = ApsPool(64, return_poly_indices=False)     
        
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1, padding_mode=padding_mode)
        self.bn4 = nn.BatchNorm2d(128)
        self.apspool4 = ApsPool(128, return_poly_indices=False)   
        
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1, padding_mode=padding_mode) 
        self.apspool5 = ApsPool(256, return_poly_indices=False)
        self.bn5 = nn.BatchNorm2d(256)

        self.conv6 = nn.Conv2d(256, 512, kernel_size=3, padding=1, padding_mode=padding_mode)  
        self.apspool6 = ApsPool(512, return_poly_indices=False)
        self.bn6 = nn.BatchNorm2d(512)
        
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, dimensions)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        #x = self.apspool1(x)
        x = self.dropout(x) 
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.apspool2(x)
        x = self.dropout(x) 
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.apspool3(x)
        x = self.dropout(x) 
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.apspool4(x)
        x = self.dropout(x) 
        
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)
        x = self.apspool5(x)
        x = self.dropout(x) 
        
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu(x)
        x = self.apspool6(x)
        x = self.dropout(x)
        
        x = torch.flatten(self.avgpool(x), 1)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)    
        
        return x

class SimpleCNN6_dropout(nn.Module):
    def __init__(self, dimensions=128, padding_mode='circular', dropout_rate=0.3):
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

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1, padding_mode=padding_mode) 
        self.lpd5 = set_pool(partial(
            PolyphaseInvariantDown2D,
            component_selection=LPS,
            get_logits=get_logits_model('LPSLogitLayers'),
            pass_extras=False
            ),p_ch=256,h_ch=256)
        self.bn5 = nn.BatchNorm2d(256)

        self.conv6 = nn.Conv2d(256, 512, kernel_size=3, padding=1, padding_mode=padding_mode)  
        self.lpd6 = set_pool(partial(
            PolyphaseInvariantDown2D,
            component_selection=LPS,
            get_logits=get_logits_model('LPSLogitLayers'),
            pass_extras=False
            ),p_ch=512,h_ch=512)
        self.bn6 = nn.BatchNorm2d(512)

        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.relu = nn.LeakyReLU(0.1)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, dimensions)  
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        #x = self.lpd1(x)
        x = self.dropout(x) 
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.lpd2(x)
        x = self.dropout(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.lpd3(x)
        x = self.dropout(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.lpd4(x)
        x = self.dropout(x)
        
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)
        x = self.lpd5(x)
        x = self.dropout(x)
        
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu(x)
        x = self.lpd6(x)
        x = self.dropout(x)
        
        x = torch.flatten(self.avgpool(x),1)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        return x    
    
class SimpleCNN6_CBAM_dropout(nn.Module):
    def __init__(self, dimensions=128, padding_mode='circular', dropout_rate=0.3):
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

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1, padding_mode=padding_mode) 
        self.lpd5 = set_pool(partial(
            PolyphaseInvariantDown2D,
            component_selection=LPS,
            get_logits=get_logits_model('LPSLogitLayers'),
            pass_extras=False
            ),p_ch=256,h_ch=256)  
        self.bn5 = nn.BatchNorm2d(256)

        self.conv6 = nn.Conv2d(256, 512, kernel_size=3, padding=1, padding_mode=padding_mode)  
        self.lpd6 = set_pool(partial(
            PolyphaseInvariantDown2D,
            component_selection=LPS,
            get_logits=get_logits_model('LPSLogitLayers'),
            pass_extras=False
            ),p_ch=512,h_ch=512)  
        self.bn6 = nn.BatchNorm2d(512)

        self.cbam1 = CBAM(16, reduction_ratio=4)
        self.cbam2 = CBAM(32, reduction_ratio=8)
        self.cbam3 = CBAM(64, reduction_ratio=16)
        self.cbam4 = CBAM(128, reduction_ratio=32)
        self.cbam5 = CBAM(256, reduction_ratio=64)
        self.cbam6 = CBAM(512, reduction_ratio=128)       
    
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.relu = nn.LeakyReLU(0.1)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, dimensions)  
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.cbam1(x)
        x = self.relu(x)
        #x = self.lpd1(x)
        x = self.dropout(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.cbam2(x)
        x = self.relu(x)
        x = self.lpd2(x)
        x = self.dropout(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.cbam3(x)
        x = self.relu(x)
        x = self.lpd3(x)
        x = self.dropout(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.cbam4(x)
        x = self.relu(x)
        x = self.lpd4(x)
        x = self.dropout(x)
        
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.cbam5(x)
        x = self.relu(x)
        x = self.lpd5(x)
        x = self.dropout(x)
        
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.cbam6(x)
        x = self.relu(x)
        x = self.lpd6(x)
        x = self.dropout(x)
        
        x = torch.flatten(self.avgpool(x),1)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        return x
    
class SimpleCNN6_CBAM_dropout_altLPS(nn.Module):
    def __init__(self, dimensions=128, padding_mode='circular', dropout_rate=0.3):
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

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1, padding_mode=padding_mode) 
        self.lpd5 = set_pool(partial(
            PolyphaseInvariantDown2D,
            component_selection=LPS,
            get_logits=get_logits_model('LPSLogitLayers'),
            pass_extras=False
            ),p_ch=256,h_ch=256)  
        self.bn5 = nn.BatchNorm2d(256)

        self.conv6 = nn.Conv2d(256, 512, kernel_size=3, padding=1, padding_mode=padding_mode)  
        self.lpd6 = set_pool(partial(
            PolyphaseInvariantDown2D,
            component_selection=LPS,
            get_logits=get_logits_model('LPSLogitLayers'),
            pass_extras=False
            ),p_ch=512,h_ch=512)  
        self.bn6 = nn.BatchNorm2d(512)

        self.cbam1 = CBAM(16, reduction_ratio=4)
        self.cbam2 = CBAM(32, reduction_ratio=8)
        self.cbam3 = CBAM(64, reduction_ratio=16)
        self.cbam4 = CBAM(128, reduction_ratio=32)
        self.cbam5 = CBAM(256, reduction_ratio=64)
        self.cbam6 = CBAM(512, reduction_ratio=128)       
    
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.relu = nn.LeakyReLU(0.1)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, dimensions)  
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.cbam1(x)
        x = self.relu(x)
        #x = self.lpd1(x)
        x = self.dropout(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.cbam2(x)
        x = self.relu(x)
        x = self.lpd2(x)
        x = self.dropout(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.cbam3(x)
        x = self.relu(x)
        #x = self.lpd3(x)
        x = self.dropout(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.cbam4(x)
        x = self.relu(x)
        x = self.lpd4(x)
        x = self.dropout(x)
        
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.cbam5(x)
        x = self.relu(x)
        #x = self.lpd5(x)
        x = self.dropout(x)
        
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.cbam6(x)
        x = self.relu(x)
        x = self.lpd6(x)
        x = self.dropout(x)
        
        x = torch.flatten(self.avgpool(x),1)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        return x  
    

class SimpleCNN4_dropout(nn.Module):
    def __init__(self, dimensions=128, padding_mode='circular', dropout_rate=0.3):
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
    
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.relu = nn.LeakyReLU(0.1)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(128, dimensions)  
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.lpd1(x)
        x = self.dropout(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.lpd2(x)
        x = self.dropout(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.lpd3(x)
        x = self.dropout(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.lpd4(x)
        x = self.dropout(x)
        
        x = torch.flatten(self.avgpool(x),1)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        return x
    
class SimpleCNN4_CBAM_dropout(nn.Module):
    def __init__(self, dimensions=128, padding_mode='circular', dropout_rate=0.3):
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

        self.cbam1 = CBAM(16, reduction_ratio=4)
        self.cbam2 = CBAM(32, reduction_ratio=8)
        self.cbam3 = CBAM(64, reduction_ratio=16)
        self.cbam4 = CBAM(128, reduction_ratio=32)

        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.relu = nn.LeakyReLU(0.1)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(128, dimensions)  
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.cbam1(x)
        x = self.relu(x)
        x = self.lpd1(x)
        x = self.dropout(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.cbam2(x)
        x = self.relu(x)
        x = self.lpd2(x)
        x = self.dropout(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.cbam3(x)
        x = self.relu(x)
        x = self.lpd3(x)
        x = self.dropout(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.cbam4(x)
        x = self.relu(x)
        x = self.lpd4(x)
        x = self.dropout(x)
        
        x = torch.flatten(self.avgpool(x),1)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        return x
    
class SimpleCNN4_dropout_2fc(nn.Module):
    def __init__(self, dimensions=128, padding_mode='circular', dropout_rate=0.3):
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
    
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.relu = nn.LeakyReLU(0.1)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(128, 256)
        self.ln1 = nn.LayerNorm(256)
        self.fc2 = nn.Linear(256, dimensions)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.lpd1(x)
        x = self.dropout(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.lpd2(x)
        x = self.dropout(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.lpd3(x)
        x = self.dropout(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.lpd4(x)
        x = self.dropout(x)
        
        #should do bn --> relu --> drop
        x = torch.flatten(self.avgpool(x),1)
        x = self.fc1(x)
        x = self.ln1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.normalize(x, p=2, dim=1)
        return x
    
class SimpleCNN4_aps_dropout_2fc(nn.Module):
    def __init__(self, dimensions=10, padding_mode='circular', dropout_rate=0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1, padding_mode=padding_mode)
        self.bn1 = nn.BatchNorm2d(16)
        self.apspool1 = ApsPool(16, return_poly_indices=False)
        self.relu = nn.LeakyReLU(0.1)  
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1, padding_mode=padding_mode)
        self.bn2 = nn.BatchNorm2d(32)
        self.apspool2 = ApsPool(32, return_poly_indices=False)     
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1, padding_mode=padding_mode)
        self.bn3 = nn.BatchNorm2d(64)
        self.apspool3 = ApsPool(64, return_poly_indices=False)     
        
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1, padding_mode=padding_mode)
        self.bn4 = nn.BatchNorm2d(128)
        self.apspool4 = ApsPool(128, return_poly_indices=False)     
        
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(128, 256)
        self.ln1 = nn.LayerNorm(256)
        self.fc2 = nn.Linear(256, dimensions)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.apspool1(x)
        x = self.dropout(x) 
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.apspool2(x)
        x = self.dropout(x) 
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.apspool3(x)
        x = self.dropout(x) 
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.apspool4(x)
        x = self.dropout(x) 
        
        x = torch.flatten(self.avgpool(x), 1)
        x = self.fc1(x)
        x = self.ln1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.normalize(x, p=2, dim=1)
        
        return x    
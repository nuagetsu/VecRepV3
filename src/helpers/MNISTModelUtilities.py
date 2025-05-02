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

#####################################################################
class SimpleCNN4(nn.Module):
    def __init__(self, dimensions=128, padding_mode='circular', dropout_rate=0.3, num_classes=10):
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

        self.relu = nn.LeakyReLU(0.1)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(128, dimensions) 
        
        self.classifier = nn.Linear(dimensions, num_classes)
        
    def forward(self, x):
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
        
        embedding = x  

        logits = self.classifier(x*10) 
        return logits, embedding
    
class SimpleCNN4_dropout(nn.Module):
    def __init__(self, dimensions=128, padding_mode='circular', dropout_rate=0.3, num_classes=10):
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

        self.relu = nn.LeakyReLU(0.1)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(128, dimensions) 
        self.dropout = nn.Dropout2d(p=dropout_rate)
        
        self.classifier = nn.Linear(dimensions, num_classes)
        
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
        
        embedding = x  

        logits = self.classifier(x*10) 
        return logits, embedding    
    
class SimpleCNN6(nn.Module):
    def __init__(self, dimensions=128, padding_mode='circular', dropout_rate=0.3, num_classes=10):
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
        self.classifier = nn.Linear(dimensions, num_classes)
        
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
                
        embedding = x  

        logits = self.classifier(x*10) 
        return logits, embedding    
    
class SimpleCNN6_dropout(nn.Module):
    def __init__(self, dimensions=128, padding_mode='circular', dropout_rate=0.3, num_classes=10):
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
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.fc = nn.Linear(512, dimensions) 
        self.classifier = nn.Linear(dimensions, num_classes)
        
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
        
        embedding = x  

        logits = self.classifier(x*10) 
        return logits, embedding      
#########################################################################
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
    
class SimpleCNN4_CBAM(nn.Module):
    def __init__(self, dimensions=128, padding_mode='circular', dropout_rate=0.3, num_classes=10):
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

        self.relu = nn.LeakyReLU(0.1)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(128, dimensions) 
        
        self.classifier = nn.Linear(dimensions, num_classes)
#         nn.init.normal_(self.classifier.weight, std=0.01)
#         nn.init.zeros_(self.classifier.bias)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.cbam1(x)
        x = self.relu(x)
        x = self.lpd1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.cbam2(x)
        x = self.relu(x)
        x = self.lpd2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.cbam3(x)
        x = self.relu(x)
        x = self.lpd3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.cbam4(x)
        x = self.relu(x)
        x = self.lpd4(x)
        
        x = torch.flatten(self.avgpool(x),1)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        
        embedding = x  

        logits = self.classifier(x*10) 
        return logits, embedding
    
class SimpleCNN4_CBAM_dropout(nn.Module):
    def __init__(self, dimensions=128, padding_mode='circular', dropout_rate=0.3, num_classes=10):
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

        self.relu = nn.LeakyReLU(0.1)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(128, dimensions) 
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.classifier = nn.Linear(dimensions, num_classes)

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
        
        embedding = x  

        logits = self.classifier(x*10) 
        return logits, embedding    
    
class SimpleCNN6_CBAM(nn.Module):
    def __init__(self, dimensions=128, padding_mode='circular', dropout_rate=0.3, num_classes=10):
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
        self.classifier = nn.Linear(dimensions, num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.cbam1(x)
        x = self.relu(x)
        #x = self.lpd1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.cbam2(x)
        x = self.relu(x)
        x = self.lpd2(x)
   
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.cbam3(x)
        x = self.relu(x)
        x = self.lpd3(x)
    
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.cbam4(x)
        x = self.relu(x)
        x = self.lpd4(x)
        
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.cbam5(x)
        x = self.relu(x)
        x = self.lpd5(x)
       
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.cbam6(x)
        x = self.relu(x)
        x = self.lpd6(x)
     
        x = torch.flatten(self.avgpool(x),1)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        
        embedding = x  

        logits = self.classifier(x*10) 
        return logits, embedding  
    
class SimpleCNN6_CBAM_dropout(nn.Module):
    def __init__(self, dimensions=128, padding_mode='circular', dropout_rate=0.3, num_classes=10):
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
        self.classifier = nn.Linear(dimensions, num_classes)
        
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
        
        embedding = x  

        logits = self.classifier(x*10) 
        return logits, embedding  
    
##################################################################33   
class SimpleCNN4_aps(nn.Module):
    def __init__(self, dimensions=10, padding_mode='circular', dropout_rate=0.3, num_classes=10):
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
        self.classifier = nn.Linear(dimensions, num_classes)
        
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
        
        embedding = x  

        logits = self.classifier(x*10) 
        return logits, embedding  
    
class SimpleCNN6_aps(nn.Module):
    def __init__(self, dimensions=128, padding_mode='circular', dropout_rate=0.3, num_classes=10):
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
        self.classifier = nn.Linear(dimensions, num_classes)
        
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
        
        embedding = x  

        logits = self.classifier(x*10) 
        return logits, embedding
import sys
import os
path = os.path.abspath("../VecRepV3") 
sys.path.append(path)

import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Sampler, random_split, Dataset
from torchvision import datasets, transforms
import torchvision
import matplotlib.pyplot as plt
from line_profiler import profile
from itertools import combinations
from sklearn.model_selection import train_test_split
import numpy as np
import random
from functools import partial
import gc
from tqdm import tqdm

import src.visualization.Metrics as metrics
import src.helpers.ModelUtilities as models
import src.data_processing.ImageProducts as ImageProducts
import src.data_processing.ImageCalculations as imgcalc

from learnable_polyphase_sampling.learn_poly_sampling.layers import get_logits_model, PolyphaseInvariantDown2D, LPS
from learnable_polyphase_sampling.learn_poly_sampling.layers.polydown import set_pool

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

imageType = "MNIST"
# ---------------------------------- Seed --------------------------------------
def set_seed(seed=42):
    random.seed(seed)  
    np.random.seed(seed)  
    torch.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False  

set_seed(42)
# ----------------------------------Input Images----------------------------------
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform)

# ----------------------------------Preparing the Dataset----------------------------------
train_size = int(0.8 * len(trainset))
val_size = len(trainset) - train_size
train_dataset, val_dataset = random_split(trainset, [train_size, val_size])
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=64, shuffle=False)

# ----------------------------------Model Architecture----------------------------------
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
    
# ----------------------------------Training Settings----------------------------------
model = SimpleCNN4_CBAM().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
num_epochs = 30
best_val_acc = 0.0
# ---------------------------------- Training Loop ----------------------------------
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(dataloader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits, _ = model(images)  # Ignore embeddings for now
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(logits, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation", leave=False):
            images, labels = images.to(device), labels.to(device)
            logits, _ = model(images)

            loss = criterion(logits, labels)
            running_loss += loss.item() * images.size(0)

            _, preds = torch.max(logits, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")

    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc = validate(model, val_loader, criterion, device)

    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'model/best_MNIST_model_classification_noini.pt')
        print("Best model saved.")
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

import matplotlib.pyplot as plt
from line_profiler import profile
from itertools import combinations
from sklearn.model_selection import train_test_split
import numpy as np
import random
from functools import partial

from src.data_processing.SampleEstimator import SampleEstimator
import src.visualization.BFmethod as graphing
import src.visualization.Metrics as metrics
import src.data_processing.ImageProducts as ImageProducts
import src.data_processing.ImageCalculations as imgcalc

from learnable_polyphase_sampling.learn_poly_sampling.layers import get_logits_model, PolyphaseInvariantDown2D, LPS
from learnable_polyphase_sampling.learn_poly_sampling.layers.polydown import set_pool

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

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
IMAGE_TYPES = ["NbinMmax_ones", "Nbin", "triangles", "triangle_mean_subtracted"]

IMAGE_FILTERS = ["unique", "Nmax_ones", "one_island"]

IMAGE_PRODUCT_TYPES = ["ncc", "ncc_scaled"]

EMBEDDING_TYPES = ["pencorr_D"]

dimensions = 32

imageType = "shapes_3_dims_24_4"
k=5
# ----------------------------------Preparing the Dataset----------------------------------
training_dataset = np.load("data/train_images_32x32.npy")
full_dataset = training_dataset #[] to generate translation
# for i in training_dataset:
#     full_dataset.append(i)
#     translated_images = imgcalc.generate_all_translations(i)
#     for j in translated_images:
#         full_dataset.append(j)
        
class CustomDataset(Dataset):
    def __init__(self, input_data):
        self.data = input_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def custom_collate(batch):
    batch_data, batch_indices = zip(*batch) 
    batch_data = torch.stack(batch_data)  
    batch_indices = torch.tensor(batch_indices)  
    return batch_data, batch_indices  

input_dataset = []
for i in full_dataset:
    img = np.array(i, dtype=np.float64)
    img = torch.from_numpy(img)
    img = img.unsqueeze(0).unsqueeze(0).cuda().double()  #1x1xHxW

    # 3 channel for RGB like input
    # img = img.repeat(1, 3, 1, 1)  #1x3xHxW
    input_dataset.append((img, i))

images, indices = zip(*input_dataset)  
stacked_images = torch.stack(images)  
stacked_images = stacked_images.cpu().numpy()
tensor_dataset = [(torch.tensor(img), idx) for img, idx in zip(stacked_images, indices)]

batch_size = 32

dataset = CustomDataset(tensor_dataset)
print(len(dataset))
train_size = int(0.8 * len(dataset))  # 80% for training
test_size = len(dataset) - train_size  # 20% for testing
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate, drop_last=True)

test_dataloader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate, drop_last=True)

# ----------------------------------Model Architecture----------------------------------
class SimpleCNN1(nn.Module):
    def __init__(self, dimensions=10, padding_mode='circular'):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1, padding_mode=padding_mode)
        self.lpd = set_pool(partial(
            PolyphaseInvariantDown2D,
            component_selection=LPS,
            get_logits=get_logits_model('LPSLogitLayers'),
            pass_extras=False
            ),p_ch=16,h_ch=16)

        self.bn1   = nn.BatchNorm2d(16)
        self.relu = nn.LeakyReLU(0.1)
             
        self.maxpool = nn.MaxPool2d(2)
        self.avgpool=nn.AdaptiveAvgPool2d((1,1))
        self.fc=nn.Linear(16, dimensions)
        
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.lpd(x)  # Use just as any down-sampling layer!
        x = torch.flatten(self.avgpool(x),1)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        return x
    
class SimpleCNN2(nn.Module):
    def __init__(self, dimensions=10, padding_mode='circular'):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1, padding_mode=padding_mode)
        self.lpd = set_pool(partial(
            PolyphaseInvariantDown2D,
            component_selection=LPS,
            get_logits=get_logits_model('LPSLogitLayers'),
            pass_extras=False
            ),p_ch=32,h_ch=32)

        self.bn1   = nn.BatchNorm2d(16)
        self.relu = nn.LeakyReLU(0.1)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1, padding_mode=padding_mode)
        self.bn2   = nn.BatchNorm2d(32)
        self.maxpool = nn.MaxPool2d(2)
        self.avgpool=nn.AdaptiveAvgPool2d((1,1))
        self.fc=nn.Linear(32, dimensions)
        
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.lpd(x)  # Use just as any down-sampling layer!
        x = torch.flatten(self.avgpool(x),1)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        return x
    
class SimpleCNN3(nn.Module):
    def __init__(self, dimensions=10, padding_mode='circular'):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1, padding_mode=padding_mode)
        self.lpd = set_pool(partial(
            PolyphaseInvariantDown2D,
            component_selection=LPS,
            get_logits=get_logits_model('LPSLogitLayers'),
            pass_extras=False
            ),p_ch=64,h_ch=64)

        self.bn1   = nn.BatchNorm2d(16)
        self.relu = nn.LeakyReLU(0.1)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1, padding_mode=padding_mode)
        self.bn2   = nn.BatchNorm2d(32)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1, padding_mode=padding_mode)
        self.bn3   = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(2)
        self.avgpool=nn.AdaptiveAvgPool2d((1,1))
        self.fc=nn.Linear(64, dimensions)
        
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.lpd(x)  # Use just as any down-sampling layer!
        x = torch.flatten(self.avgpool(x),1)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        return x

# ----------------------------------Training Settings----------------------------------
# def loss_fn(A, G):
#     return torch.norm(A - G, p='fro')  

def loss_fn(A,G):
    return F.mse_loss(A, G)

# -------------------------------- Loop over different dimensions and models--------------------------
dimensions = [8, 16, 32, 64, 128, 256]

models = [SimpleCNN1, SimpleCNN2, SimpleCNN3]
# ---------------------------------- Training Loop ----------------------------------
for i, model_class in enumerate(models):
    for dimension in dimensions:
        print(f"Training {model_class.__name__} with conv layer of {i+1} and dimension {dimension}")

        model = model_class(dimensions=dimension, padding_mode='circular').to(device)
        train_loss_history = []
        val_loss_history = []
        
        optimizer = optim.Adam(model.parameters(), lr=0.0001)

        epochs = 20
        plot_epoch = epochs
        patience = 3
        best_val_loss = float('inf')
        epochs_no_improve = 0

        for epoch in range(epochs):
            model.train()
            training_loss, total_loss_training = 0, 0
            for batch_data, batch_indices in train_dataloader: #3500
                optimizer.zero_grad()
                loss_per_pair = 0
                len_train = 0
                remaining_indices = list(range(len(batch_data)))
                for idx1, idx2 in combinations(remaining_indices, 2): #16C2
                    data1, data2 = batch_data[idx1], batch_data[idx2]

                    img1 = data1.cuda().float()
                    img2 = data2.cuda().float()

                    embedded_vector_image1 = model(img1)
                    embedded_vector_image2 = model(img2)

                    dot_product_value = torch.sum(embedded_vector_image1 * embedded_vector_image2, dim=1) 
                    scale = ImageProducts.scale_min(ImageProducts.ncc, -1)

                    input1 = data1.squeeze(0)[0].cpu().numpy()
                    input2 = data2.squeeze(0)[0].cpu().numpy()
                    NCC_scaled_value = scale(input1,input2)
                    NCC_scaled_value = torch.tensor(NCC_scaled_value).to(dot_product_value.device).float()
                    if NCC_scaled_value.ndim == 0:
                        NCC_scaled_value = NCC_scaled_value.unsqueeze(0)

                    loss = loss_fn(dot_product_value, NCC_scaled_value) #squared frobenius norm 
                    loss_per_pair += loss
                    len_train += 1

                training_loss = loss_per_pair/len_train
                print(f"training_loss in epoch {epoch}: {training_loss}")

                training_loss.backward()
                optimizer.step()

                total_loss_training += training_loss.item()  

            avg_loss = total_loss_training /  (len(train_dataloader))
            train_loss_history.append(avg_loss)
            print(f"\nEpoch {epoch}: Avg Loss = {avg_loss:.4f}")

            # Validation loop
            model.eval()
            validation_loss, total_loss_validation = 0, 0        
            with torch.no_grad():
                for batch_data, batch_indices in test_dataloader:  
                    loss_per_pair = 0
                    len_test = 0
                    remaining_indices = list(range(len(batch_data))) 
                    for idx1, idx2 in combinations(remaining_indices, 2):
                        data1, data2 = batch_data[idx1], batch_data[idx2]

                        img1 = data1.cuda().float()
                        img2 = data2.cuda().float()

                        embedded_vector_image1 = model(img1)
                        embedded_vector_image2 = model(img2)

                        dot_product_value = torch.sum(embedded_vector_image1 * embedded_vector_image2, dim=1)

                        scale = ImageProducts.scale_min(ImageProducts.ncc, -1)
                        input1 = data1.squeeze(0)[0].cpu().numpy()
                        input2 = data2.squeeze(0)[0].cpu().numpy()
                        NCC_scaled_value = scale(input1,input2)
                        NCC_scaled_value = torch.tensor(NCC_scaled_value).to(dot_product_value.device).float()
                        if NCC_scaled_value.ndim == 0:
                            NCC_scaled_value = NCC_scaled_value.unsqueeze(0)

                        loss = loss_fn(dot_product_value, NCC_scaled_value)

                        loss_per_pair += loss.item()
                        len_test +=1

                    validation_loss = loss_per_pair/len_test
                    total_loss_validation += validation_loss
                    print("validation_loss: ", validation_loss)

                avg_val_loss = total_loss_validation / (len(test_dataloader))
                val_loss_history.append(avg_val_loss)

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    epochs_no_improve = 0
                    torch.save(model.state_dict(), f'model/best_model_{imageType}_{dimension}d_convlayer{i+1}.pt')
                else:
                    epochs_no_improve += 1

                # Early stopping
                if epochs_no_improve == patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    plot_epoch = epoch+1
                    break
                #torch.save(model.state_dict(), f'model/best_model_{imageType}_{dimension}d.pt')
                print(f"Epoch {epoch}: Validation Loss: {avg_val_loss:.4f}")

        # ----------------------------------Plots----------------------------------
        plt.figure()
        plt.plot(train_loss_history, label="Train Loss")
        plt.plot(val_loss_history, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.savefig(f"model/loss_{imageType}_{dimension}d_convlayer{i+1}.png")    


        with open("model/output_3.txt", "a") as file:
            file.write(f"best_model_{imageType}_{dimension}d_convlayer{i+1}\n")
            for item in val_loss_history:
                file.write(f"{item}\n")
    

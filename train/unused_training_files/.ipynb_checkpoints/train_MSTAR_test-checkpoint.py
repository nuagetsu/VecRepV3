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

import src.visualization.Metrics as metrics
import src.helpers.ModelUtilities as models
import src.data_processing.ImageProducts as ImageProducts
import src.data_processing.ImageCalculations as imgcalc

from learnable_polyphase_sampling.learn_poly_sampling.layers import get_logits_model, PolyphaseInvariantDown2D, LPS
from learnable_polyphase_sampling.learn_poly_sampling.layers.polydown import set_pool

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

imageType = "MSTAR"
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
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])

data_dir = 'data/mstar-dataset-8-classes/Padded_imgs'  

# ----------------------------------Preparing the Dataset----------------------------------    
class CustomDataset(Dataset):
    def __init__(self, file_path, max_samples=150000):
        self.data = np.load(file_path, mmap_mode='r')  
        self.data = self.data[:max_samples]  

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = np.array(self.data[idx], dtype=np.float32)  #use float32 instead of float64
        img = torch.from_numpy(img)
        img = img.unsqueeze(0).unsqueeze(0)
        return img, idx  

dataset = CustomDataset(data_dir)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

batch_size = 32 #the max gpu memory can handle

def custom_collate(batch):
    batch_data, batch_indices = zip(*batch)
    batch_data = torch.stack(batch_data)
    batch_indices = torch.tensor(batch_indices)
    return batch_data, batch_indices

train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate, drop_last=True, num_workers=4, pin_memory=True)

print(len(dataset),len(train_dataloader))
test_dataloader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate, drop_last=True, num_workers=4, pin_memory=True)


print("len(train_dataloader): ",len(train_dataloader)) 
#Total images: 9469
#Training images: 5681, Validation images: 1893, Test images: 1895

# ----------------------------------Model Architecture----------------------------------
class BasicBlockLPS(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlockLPS, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=1, 
            padding=1, bias=False, padding_mode='circular'
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1,
            padding=1, bias=False, padding_mode='circular'
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNetLPS(nn.Module):
    def __init__(self, block, layers, embedding_dim=512, num_classes=None):
        super(ResNetLPS, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(
            1, 64, kernel_size=3, stride=1,
            padding=1, bias=False, padding_mode='circular'
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # LPS downsampling
        self.lps_pool = set_pool(partial(
            PolyphaseInvariantDown2D,
            component_selection=LPS,
            get_logits=get_logits_model('LPSLogitLayers'),
            pass_extras=False
        ), p_ch=64, h_ch=64)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.embedding = nn.Linear(512 * block.expansion, embedding_dim)
        nn.init.normal_(self.embedding.weight, 0, 0.01)  
        
        self.classifier = nn.Linear(embedding_dim, num_classes) if num_classes else None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = set_pool(partial(
                PolyphaseInvariantDown2D,
                component_selection=LPS,
                get_logits=get_logits_model('LPSLogitLayers'),
                pass_extras=False
            ), p_ch=self.in_planes, h_ch=planes * block.expansion)

        layers = []
        layers.append(block(self.in_planes, planes, 1, downsample))
        self.in_planes = planes * block.expansion
        
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.lps_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        features = torch.flatten(self.avgpool(x), 1)
        embeddings = F.normalize(self.embedding(features), p=2, dim=1)
        
        #probably not needed anyways
        if self.classifier is not None:
            logits = self.classifier(embeddings)
            return logits, embeddings
            
        return embeddings

def ResNet18LPS(embedding_dim=512, num_classes=None):
    return ResNetLPS(BasicBlockLPS, [2,2,2,2], 
                    embedding_dim=embedding_dim, 
                    num_classes=num_classes).to(device)


ResNet18 = ResNet18LPS(embedding_dim=256).to(device)

# ----------------------------------Training Settings----------------------------------
def loss_fn(A,G):
    return F.mse_loss(A, G)
    
# -------------------------------- Loop over different dimensions and models--------------------------
dimensions = [512, 1028]

model_class = [ResNet18]
# -------------------------------- Loop over different dimensions and models--------------------------
for i, model_class in enumerate(model_class):
    for dimension in dimensions:
        print(f"Training MSTAR {model_class.__name__} with conv layer of {i+18} and dimension {dimension}")
        with open("model/output_8.txt", "a", buffering=1) as file_model:
            file_model.write(f"\nTraining {model_class.__name__} with conv layer of {i+18} and dimension {dimension}")

        model = model_class(embedding_dim=dimension).to(device)
        train_loss_history = []
        val_loss_history = []

        optimizer = optim.Adam(model.parameters(), lr=0.0005)

        epochs = 50
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
                    NCC_scaled_value = NCC_scaled_value.clamp(min=-1.0)

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
            with open("model/output_8.txt", "a", buffering=1) as file_model:
                file_model.write(f"\nEpoch {epoch}: Avg Loss = {avg_loss:.4f}, {model_class.__name__}, {imageType}")

            # Clear Cache
            torch.cuda.empty_cache()             
            for var in ["batch_data", "batch_indices", "training_loss", "total_loss_training"]:
                if var in locals():
                    del locals()[var]

            gc.collect()

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
                        NCC_scaled_value = NCC_scaled_value.clamp(min=-1.0)
                        
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
                    torch.save(model.state_dict(), f'model/best_model_MSTAR_{imageType}_{dimension}d_convlayer{i+18}_{model_class.__name__}.pt')
                else:
                    epochs_no_improve += 1

                # Early stopping
                if epochs_no_improve == patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    plot_epoch = epoch+1
                    break
                #torch.save(model.state_dict(), f'model/best_model_{imageType}_{dimension}d.pt')
                print(f"Epoch {epoch}: Validation Loss: {avg_val_loss:.4f}")
                with open("model/output_8.txt", "a", buffering=1) as file_model:
                    file_model.write(f"\nEpoch {epoch}: Validation Loss: {avg_val_loss:.4f}, {model_class.__name__}, {imageType}")

            # Clear Cache
            torch.cuda.empty_cache()             
            for var in ["batch_data", "batch_indices", "validation_loss", "total_loss_validation"]:
                if var in locals():
                    del locals()[var]

            gc.collect()

    # ----------------------------------Plots----------------------------------
        plt.figure()
        plt.plot(train_loss_history, label="Train Loss")
        plt.plot(val_loss_history, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.savefig(f"model/loss_MSTAR_{imageType}_{dimension}d_convlayer{i+3}_{model_class.__name__}_{imageType}.png")    


        with open("model/output_MSTAR.txt", "a") as file:
            file.write(f"best_model_MSTAR_{imageType}_{dimension}d_convlayer{i+3}_{model_class.__name__}_{imageType}\n")
            for item in val_loss_history:
                file.write(f"{item}\n")   
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

imageType = "FashionMNIST"
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

aug_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomRotation(degrees=20),           
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  
    transforms.RandomHorizontalFlip(),        
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, transform=transform)
# augmented_trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=aug_transform)

# full_dataset = torch.utils.data.ConcatDataset([trainset, augmented_trainset])
full_dataset = trainset
# ----------------------------------Preparing the Dataset----------------------------------
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

print(f"\nGenerating input dataset")
input_dataset = []
for i in range(len(full_dataset)):
    image,label = full_dataset[i]
    img = image.unsqueeze(0).cuda().double() 
    input_dataset.append((img, i))


images, indices = zip(*input_dataset)  
stacked_images = torch.stack(images)  
stacked_images = stacked_images.cpu().numpy()
tensor_dataset = [(torch.tensor(img), idx) for img, idx in zip(stacked_images, indices)]

batch_size = 32

dataset = CustomDataset(tensor_dataset)

train_size = int(0.8 * len(dataset))  # 80% for training
test_size = len(dataset) - train_size  # 20% for testing
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate, drop_last=True)

test_dataloader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate, drop_last=True)

print("len(train_dataloader): ",len(train_dataloader)) 
# ----------------------------------Model Architecture----------------------------------
SimpleCNN6 = models.SimpleCNN6 #done
SimpleCNN4 = models.SimpleCNN4
SimpleCNN2 = models.SimpleCNN2

#dropout does not seem to work
SimpleCNN4_CBAM_dropout = models.SimpleCNN4_CBAM_dropout
SimpleCNN4_2fc_dropout = models.SimpleCNN4_dropout_2fc
# ----------------------------------Training Settings----------------------------------
def loss_fn(A,G):
    return F.mse_loss(A, G)
# -------------------------------- Loop over different dimensions and models--------------------------
dimensions = [64, 128]

model_class = [SimpleCNN2]
# ----------------------------------Training Loop----------------------------------
for i, model_class in enumerate(model_class):
    for dimension in dimensions:
        print(f"Training MNIST {model_class.__name__} with conv layer of {i+3} and dimension {dimension}")
        with open("model/output_7.txt", "a", buffering=1) as file_model:
            file_model.write(f"\nTraining {model_class.__name__} with conv layer of {i+3} and dimension {dimension}")

        model = model_class(dimensions=dimension, padding_mode='circular').to(device)
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
            with open("model/output_7.txt", "a", buffering=1) as file_model:
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
                    torch.save(model.state_dict(), f'model/best_model_MNIST_{imageType}_{dimension}d_convlayer{i+3}_{model_class.__name__}.pt')
                else:
                    epochs_no_improve += 1

                # Early stopping
                if epochs_no_improve == patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    plot_epoch = epoch+1
                    break
                #torch.save(model.state_dict(), f'model/best_model_{imageType}_{dimension}d.pt')
                print(f"Epoch {epoch}: Validation Loss: {avg_val_loss:.4f}")
                with open("model/output_7.txt", "a", buffering=1) as file_model:
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
        plt.savefig(f"model/loss_MNIST_{imageType}_{dimension}d_convlayer{i+3}_{model_class.__name__}_{imageType}.png")    


        with open("model/output_MNIST.txt", "a") as file:
            file.write(f"best_model_MNIST_{imageType}_{dimension}d_convlayer{i+3}_{model_class.__name__}_{imageType}\n")
            for item in val_loss_history:
                file.write(f"{item}\n")   

    

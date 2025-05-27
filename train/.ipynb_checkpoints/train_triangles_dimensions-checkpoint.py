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
import gc

import src.visualization.Metrics as metrics
import src.helpers.ModelUtilities as models
import src.helpers.oldModelUtilities as oldmodels
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

imageType = "shapes_3_dims_48_4"
k=5
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

dataset = CustomDataset("data/train_images_56x56_1.npy")

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

# ----------------------------------Model Architecture----------------------------------
SimpleCNN4_aps = models.SimpleCNN4_aps
SimpleCNN6_aps = models.SimpleCNN6_aps

SimpleCNN4_aps_dropout = models.SimpleCNN4_aps_dropout
SimpleCNN4_aps_dropout_2fc = models.SimpleCNN4_aps_dropout_2fc

SimpleCNN4 = models.SimpleCNN4
SimpleCNN4_2fc = models.SimpleCNN4_2fc
SimpleCNN4_CBAM = models.SimpleCNN4_CBAM

oldSimpleCNN4 = oldmodels.SimpleCNN4
SimpleCNN6 = models.SimpleCNN6

SimpleCNN6_dropout = models.SimpleCNN6_dropout
oldSimpleCNN6 = oldmodels.SimpleCNN6

SimpleCNN4_aps_CBAM_dropout = models.SimpleCNN4_aps_CBAM_dropout
SimpleCNN6_CBAM = models.SimpleCNN6_CBAM
SimpleCNN6_CBAM_dropout = models.SimpleCNN6_CBAM_dropout
SimpleCNN6_aps_CBAM_dropout = models.SimpleCNN6_aps_CBAM_dropout

SimpleCNN2_aps = models.SimpleCNN2_aps
SimpleCNN2_aps_CBAM = models.SimpleCNN2_aps_CBAM
SimpleCNN2_aps_dropout = models.SimpleCNN2_aps_dropout
SimpleCNN2_aps_CBAM_dropout = models.SimpleCNN2_aps_CBAM_dropout
# ----------------------------------Training Settings----------------------------------
# def loss_fn(A, G):
#     return torch.norm(A - G, p='fro')  

def loss_fn(A,G):
    return F.mse_loss(A, G)

# -------------------------------- Loop over different dimensions and models--------------------------
dimensions = [64]

model_classes = [oldSimpleCNN4]
# ---------------------------------- Training Loop ----------------------------------
for i, model_class in enumerate(model_classes):
    for dimension in dimensions:
        print(f"Training {model_class.__name__} with conv layer of {i+3} and dimension {dimension}")
        with open("model/output_6.txt", "a", buffering=1) as file_model:
            file_model.write(f"\nTraining {model_class.__name__} with conv layer of {i+3} and dimension {dimension}, {imageType}")

        model = model_class(dimensions=dimension, padding_mode='circular').to(device)
        train_loss_history = []
        val_loss_history = []

        optimizer = optim.Adam(model.parameters(), lr=0.0005)

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
            with open("model/output_6.txt", "a", buffering=1) as file_model:
                file_model.write(f"\nEpoch {epoch}: Avg Loss = {avg_loss:.4f}, {model_class.__name__}, {imageType}, {dimension}")

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
                    torch.save(model.state_dict(), f'model/best_model_{imageType}_{dimension}d_convlayer{i+3}_{model_class.__name__}.pt')
                else:
                    epochs_no_improve += 1

                # Early stopping
                if epochs_no_improve == patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    plot_epoch = epoch+1
                    break
                #torch.save(model.state_dict(), f'model/best_model_{imageType}_{dimension}d.pt')
                print(f"Epoch {epoch}: Validation Loss: {avg_val_loss:.4f}")
                with open("model/output_6.txt", "a", buffering=1) as file_model:
                    file_model.write(f"\nEpoch {epoch}: Validation Loss: {avg_val_loss:.4f}, {model_class.__name__}, {imageType}, {dimension}")

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
        plt.savefig(f"model/loss_{imageType}_{dimension}d_convlayer{i+3}.png")    


        with open("model/output_5.txt", "a") as file:
            file.write(f"best_model_{imageType}_{dimension}d_convlayer{i+3}\n")
            for item in val_loss_history:
                file.write(f"{item}\n")


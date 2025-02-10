import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Sampler, random_split, Dataset
from torchvision import datasets, transforms
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
from line_profiler import profile
from itertools import combinations
from sklearn.model_selection import train_test_split
from functools import partial

import numpy as np
import random

import src.visualization.Metrics as metrics
import src.data_processing.ImageProducts as ImageProducts
import src.helpers.ModelUtilities as models

from learnable_polyphase_sampling.learn_poly_sampling.layers import get_logits_model, PolyphaseInvariantDown2D, LPS
from learnable_polyphase_sampling.learn_poly_sampling.layers.polydown import set_pool

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    
# ----------------------------------Input Images----------------------------------
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

augmentation_transform = transforms.Compose([
    transforms.Resize((32, 32)), 
    transforms.RandomAffine(
        degrees=0,  # No rotation
        translate=(0.1, 0.1),  # Random horizontal/vertical shifts (10% of image size)
        fill=0  # Pad with black (MNIST background)
    ),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform)
augmented_trainset = torchvision.datasets.MNIST(root='./data', train=True, transform=augmentation_transform)

full_dataset = torch.utils.data.ConcatDataset([trainset, augmented_trainset])
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

batch_size = 24

dataset = CustomDataset(tensor_dataset)

train_size = int(0.8 * len(dataset))  # 80% for training
test_size = len(dataset) - train_size  # 20% for testing
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate, drop_last=True)

test_dataloader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate, drop_last=True)
print("len(test_dataloader): ",len(test_dataloader)) #1500 -> 1,500 x 16 x 5 = 60,000 x 2 = 120,000
# ----------------------------------Model Architecture----------------------------------

print(f"\nLoading model architecture")
model = models.CNN().to(device)
print(f"Model architecture: \n{model}")
# ----------------------------------Training Settings----------------------------------

loss_fn = models.loss_fn_MSE

optimizer = optim.Adam(model.parameters(), lr=0.0001)

train_loss_history = []
val_loss_history = []

epochs = 100
plot_epoch = epochs
patience = 5
best_val_loss = float('inf')
epochs_no_improve = 0

# ----------------------------------Training Loop----------------------------------
for epoch in range(epochs):
    model.train()
    training_loss, total_loss_training = 0, 0
    for batch_data, batch_indices in train_dataloader: 
        optimizer.zero_grad()
        loss_per_pair = 0
        len_train = 0
        remaining_indices = list(range(len(batch_data))) #16
        for idx1, idx2 in combinations(remaining_indices, 2): #16C2
            data1, data2 = batch_data[idx1], batch_data[idx2]
            index1, index2 = batch_indices[idx1].item(), batch_indices[idx2].item()

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

    model.eval()
    validation_loss, total_loss_validation = 0, 0        
    with torch.no_grad():
        for batch_data, batch_indices in test_dataloader:  
            loss_per_pair = 0
            len_test = 0
            remaining_indices = list(range(len(batch_data))) # 16
            for idx1, idx2 in combinations(remaining_indices, 2): #16C2 = 120 
                data1, data2 = batch_data[idx1], batch_data[idx2]
                index1, index2 = batch_indices[idx1].item(), batch_indices[idx2].item()  

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
            #torch.save(model.state_dict(), 'model/best_model_batch_greyscale_mnistSimpleCNN.pt')
        else:
            epochs_no_improve += 1
 
        if epochs_no_improve == patience:
            print(f"Early stopping at epoch {epoch+1}")
            plot_epoch = epoch+1
            break
        torch.save(model.state_dict(), 'model/best_model_batch_greyscale_MNIST_circular1.pt')
        print(f"Validation Loss: {avg_val_loss:.4f}")
        
# ----------------------------------Plots----------------------------------
plt.figure()
plt.plot(train_loss_history, label="Train Loss")
plt.plot(val_loss_history, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.savefig("model/loss_batch_greyscale_MNIST_circular1.png")    

    
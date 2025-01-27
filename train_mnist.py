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
from sklearn.model_selection import train_test_split
import numpy as np
import random

import src.data_processing.BruteForceEstimator as bfEstimator
import src.visualization.BFmethod as graphing
import src.visualization.Metrics as metrics
import src.data_processing.ImageProducts as ImageProducts

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    
# ----------------------------------Input Images----------------------------------

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

trainset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform)
testset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform)
full_dataset = torch.utils.data.ConcatDataset([trainset, testset])

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

input_dataset = []
for i in range(len(full_dataset)):
    image,label = full_dataset[i]
    image_array = (image.numpy() > 0.5).astype(np.uint8)
    img = torch.from_numpy(image_array)
    img = img.unsqueeze(0).cuda().double()  #1x1xHxW

    # 3 channel for RGB like input
    # img = img.repeat(1, 3, 1, 1)  #1x3xHxW
    input_dataset.append((img, i))

print("len(input_dataset): ",len(input_dataset))
images, indices = zip(*input_dataset)  
stacked_images = torch.stack(images)  
stacked_images = stacked_images.cpu().numpy()
tensor_dataset = [(torch.tensor(img), idx) for img, idx in zip(stacked_images, indices)]

batch_size = 16

dataset = CustomDataset(tensor_dataset)

train_size = int(0.8 * len(dataset))  # 80% for training
test_size = len(dataset) - train_size  # 20% for testing
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate, drop_last=True)

test_dataloader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate, drop_last=True)

# ----------------------------------Model Architecture----------------------------------
class MNIST_CNN(nn.Module):
    def __init__(self, dimensions=128): 
        super(MNIST_CNN, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool1 = nn.MaxPool2d(2, 2)  
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool2 = nn.MaxPool2d(2, 2) 
        
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        
        self.relu = nn.LeakyReLU(0.1)

        self.fc1 = nn.Linear(512 * 7 * 7, 512)  
        self.fc2 = nn.Linear(512, dimensions)

    def forward(self, x):
        x = self.conv1(x)        
        x = self.bn1(x)          
        x = self.relu(x)        

        x = self.conv2(x)        
        x = self.bn2(x)          
        x = self.relu(x)        

        x = self.pool1(x)        
        
        x = self.conv3(x)        
        x = self.bn3(x)          
        x = self.relu(x)        

        x = self.conv4(x)        
        x = self.bn4(x)         
        x = self.relu(x)        

        x = self.pool2(x)       
      
        x = self.conv5(x)       
        x = self.bn5(x)        
        x = self.relu(x)       
        
        x = torch.flatten(x, start_dim=1)
  
        x = self.fc1(x)         
        x = self.relu(x)   
        x = self.fc2(x)         
        
        x = F.normalize(x, p=2, dim=1)

        return x

model = MNIST_CNN().to(device)

# ----------------------------------Training Settings----------------------------------
def loss_fn(A, G):
    return torch.norm(A - G, p='fro')  

# def loss_fn(A,G):
#     return F.mse_loss(A, G)

optimizer = optim.Adam(model.parameters(), lr=0.0001)

train_loss_history = []
val_loss_history = []

epochs = 500 
plot_epoch = epochs
patience = 3 
best_val_loss = float('inf')
epochs_no_improve = 0

# ----------------------------------Training Loop----------------------------------
for epoch in range(epochs):
    model.train()
    total_loss = 0
    total_acc = 0.0
    for batch_data, batch_indices in train_dataloader: 
        remaining_indices = list(range(len(batch_data)))
        for k in range(len(batch_data) // 2):
            idx1, idx2 = random.sample(remaining_indices, 2)
            data1, data2 = batch_data[idx1], batch_data[idx2]
            index1, index2 = batch_indices[idx1].item(), batch_indices[idx2].item()
            remaining_indices.remove(idx1)
            remaining_indices.remove(idx2)
            
            img1 = data1.cuda().float()
            img2 = data2.cuda().float()
            
            # print(f"image 1 in epoch {epoch} for k = {k}: {img1}")
            # print(f"image 2 in epoch {epoch} for k = {k}: {img2}")
            # print(f"Using images from original dataset indices: {index1}, {index2}")
            
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
            # print("loss: ",loss)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()  

    avg_loss = total_loss /  (len(train_dataloader)*(len(batch_data)/2))
    train_loss_history.append(avg_loss)
    print(f"\nEpoch {epoch}: Avg Loss = {avg_loss:.4f}")

    # Validation loop
    model.eval()
    val_loss = 0        

    with torch.no_grad():
        for batch_data, batch_indices in test_dataloader:  
            remaining_indices = list(range(len(batch_data))) 

            for _ in range(len(batch_data) // 2):
                idx1, idx2 = random.sample(remaining_indices, 2)
                data1, data2 = batch_data[idx1], batch_data[idx2]
                index1, index2 = batch_indices[idx1].item(), batch_indices[idx2].item()  

                remaining_indices.remove(idx1)
                remaining_indices.remove(idx2)

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

                # NCC_scaled_value = torch.tensor(
                #     scale(data1.squeeze(0)[0].cpu().numpy(), data2.squeeze(0)[0].cpu().numpy()), device="cuda", dtype=torch.float)
                # NCC_scaled_value = NCC_scaled_value.view(dot_product_value.shape)

                pair_loss = loss_fn(dot_product_value, NCC_scaled_value)

                val_loss += pair_loss.item()

        avg_val_loss = val_loss / (len(test_dataloader)*(len(batch_data)/2))
        val_loss_history.append(avg_val_loss)

#         if avg_val_loss < best_val_loss:
#             best_val_loss = avg_val_loss
#             epochs_no_improve = 0
#             torch.save(model.state_dict(), 'best_model_batch.pth')
#         else:
#             epochs_no_improve += 1
 
#         # Early stopping
#         if epochs_no_improve == patience:
#             print(f"Early stopping at epoch {epoch+1}")
#             plot_epoch = epoch+1
#             break
        torch.save(model.state_dict(), 'model/best_model_batch_greyscale_mnist_500E.pth')
        print(f"Validation Loss: {avg_val_loss:.4f}")

        
        
# ----------------------------------Plots----------------------------------
plt.figure()
plt.plot(train_loss_history, label="Train Loss")
plt.plot(val_loss_history, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.savefig("model/loss_batch_greyscale_mnist_500E.png")    

    
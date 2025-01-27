#---------------------------------------BFMethod Embedding Vector------------------------------------------------------
import sys
import os
abs_path = os.path.abspath("VecRepV3")
sys.path.append(abs_path)

import matplotlib.pyplot as plt
from line_profiler import profile
import numpy as np
import src.data_processing.BruteForceEstimator as bfEstimator
import src.visualization.BFmethod as graphing
import src.visualization.Metrics as metrics

IMAGE_TYPES = ["NbinMmax_ones", "Nbin", "triangles", "triangle_mean_subtracted"]

IMAGE_FILTERS = ["unique", "Nmax_ones", "one_island"]

IMAGE_PRODUCT_TYPES = ["ncc", "ncc_scaled"]

EMBEDDING_TYPES = ["pencorr_D"]

image_set_index = 1
dimensions = 5

imageType = "2bin"
filters = ["100max_ones"]
imageProductType = "ncc_scaled_-1"
overwrite = {"imgSet": False, "imgProd": False, "embedding": False}
weight = None
embeddingType = f"pencorr_{dimensions}"
k=2

bruteForceEstimator = bfEstimator.BruteForceEstimator(
    imageType=imageType, filters=filters, imageProductType=imageProductType, embeddingType=embeddingType, overwrite=overwrite)

print("Image set: \n", bruteForceEstimator.imageSet[image_set_index])

matrixG = bruteForceEstimator.matrixG
matrixA = bruteForceEstimator.matrixA
#print("Matrix A: ", matrixA)
np.set_printoptions(suppress=True, precision=10) 
print("\nBFMethod\n")
print("y_orig vector: ",matrixA[:,image_set_index])

if matrixG.ndim == 1:
    matrixG = matrixG.reshape(-1, 1)
if matrixA.ndim == 1:
    matrixA = matrixA.reshape(-1, 1)

# Approximation Method -- Squared Difference
dot_product_matrix = np.dot(matrixA.T, matrixA)
diff_matrix = (matrixG - dot_product_matrix)**2
total_sum = np.sum(diff_matrix)
# print("Sum of all elements in difference matrix:", total_sum)

# Appromixation Method -- KNN-IoU
kscores=[]
for i in range(len(matrixG)):
    vectorc=[]
    vectorb = bruteForceEstimator.matrixG[i]
    for j in range(len(matrixA[0])):
        vectorc.append(np.dot(bruteForceEstimator.matrixA[:, j], bruteForceEstimator.matrixA[:, i]))
    kscore= metrics.get_k_neighbour_score(vectorb, vectorc, k)
    kscores.append(kscore)
    # print(f"Estimating K-Score for Image {i}: K-Score = {kscore}")

#---------------------------------------LPS Embedding Vector------------------------------------------------------
import sys
import os
abs_path = os.path.abspath("learnable_polyphase_sampling/learn_poly_sampling")
sys.path.append(abs_path)

import numpy as np 
import torch 
import torch.nn as nn
from PIL import Image
from functools import partial
from layers import get_logits_model, PolyphaseInvariantDown2D, LPS
from layers.polydown import set_pool

# Define Model
# able to change the embedding dimensions 
class SimpleClassifier(nn.Module):
    def __init__(self, num_classes=dimensions,padding_mode='circular'):
        # Conv. Layer
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1, padding_mode=padding_mode) # C x H x W
        # Learnable Polyphase Downsampling Layer
        self.lpd = set_pool(partial(
            PolyphaseInvariantDown2D,
            component_selection=LPS,
            get_logits=get_logits_model('LPSLogitLayers'),
            pass_extras=False
            ),p_ch=32,h_ch=32)
        # Global Pooling + Classifier
        self.avgpool=nn.AdaptiveAvgPool2d((1,1))
        self.fc=nn.Linear(32, num_classes)
    def forward(self, x, return_LPS_embeddings=False, return_conv_embeddings=False):
        x = self.conv1(x)  
        if return_conv_embeddings:
            return torch.flatten(x)           
        x = self.lpd(x)    
        if return_LPS_embeddings:
            return torch.flatten(self.avgpool(x), 1)       
        x = torch.flatten(self.avgpool(x), 1)  # Global pooling
        return self.fc(x)  # Classifier output, but perhaps we can use this for dimensionality reduction...?

    
    # Construct Model
torch.manual_seed(0)
model = SimpleClassifier().cuda().eval().double()

# weights = model.conv1.weight 
# biases = model.conv1.bias  
# print("weight: ",weights)
# print("biases: ",biases)

# Load Image Batch Size(N)×Channel×Height×Width
img = np.array(bruteForceEstimator.imageSet[image_set_index], dtype=np.float64)

img = torch.from_numpy(img)
img = img.unsqueeze(0).unsqueeze(0).cuda().double()  #1x1x2x2

# Make it 3 channel for RGB like input for [0,1]
img = img.repeat(1, 3, 1, 1)  #1x3x2x2
print("\nIMG\n")
print("img: ", img)

# output embedding vector
y_orig = model(img,return_LPS_embeddings=True).detach().cpu()
print("\nAfter LPS layer\n")
print("y_orig vector: ",y_orig.numpy())

y_orig_conv = model(img,return_conv_embeddings=True).detach().cpu()
print("\nBefore LPS layer -- After conv layer\n")
print("y_orig_conv vector: ",y_orig_conv.numpy())

y = model(img).detach().cpu()
print("\nLogits vector for classification\n")
print("output logits vector: ",y.numpy())

print("Comparing dot vector")
# img_roll = torch.roll(img,shifts=(1, 1), dims=(-1, -2))
# y_roll = model(img_roll,return_LPS_embeddings=True).detach().cpu()
# print("y_roll vector: ",y_roll.numpy())
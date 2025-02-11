# **Train**
This file contains script to train different model to learn an embedding space for image similarity. The model uses **Polyphase Invariant Downsampling (LPS)** and **Normalized Cross-Correlation (NCC)** to generate embedding.


## **Script Breakdown**
### **1. Data Preparation**
- `train_triangles.py` uses `BruteForceEstimator` to load and preprocess triangle 'images'.
- Created custom dataset and dataloaders for training and validation.

### **2. Model Architecture**
- Different **CNN** model is implemented for experimentation and testing, architectures can be found in `ModelUtilities.py`.
- **Polyphase Invariant Downsampling (LPS)** is integrated for shift invariance.
- Used a fully connected layer to generate embeddings of desired dimension, followed by a L2 normalisation layer to project embeddings onto a unit hypersphere.

### **3. Loss Function**
- Uses **Mean Squared Error (MSE)** loss to compare dot product values of embeddings with NCC similarity scores.

### **4. Training Process**
- Computes embeddings for each pair of images in a batch.
- Uses NCC similarity as ground truth.

## **Output**
- Trained model and loss curve are saved in `model` file.

## **Notes**
- Model architecture and dataset parameters are still being tested


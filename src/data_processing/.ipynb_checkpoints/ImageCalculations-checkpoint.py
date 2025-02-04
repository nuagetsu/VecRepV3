import torch
import numpy as np

import src.data_processing.BruteForceEstimator as bfEstimator
import src.data_processing.ImageCalculations as imgcalc
import src.visualization.BFmethod as graphing
import src.visualization.Metrics as metrics
import src.visualization.ImagePlots as imgplt
import src.data_processing.ImageProducts as ImageProducts
import src.helpers.ModelUtilities as models


def check_translationally_unique(img1: np.ndarray, img2: np.ndarray) -> bool:
    """Check if two binary images are translationally unique."""
    if img1.shape != img2.shape:
        return True  

    squareLength = img1.shape[0]  
    all_permutations = set()  
    for dr in range(squareLength):  # Vertical shifts
        shifted_matrix = np.roll(img1, dr, axis=0)
        for dc in range(squareLength):  # Horizontal shifts
            shifted_matrix = np.roll(shifted_matrix, 1, axis=1)  # Shift right
            to_store = tuple(shifted_matrix.flatten())  
            all_permutations.add(to_store)  

            if np.array_equal(shifted_matrix, img2):
                return False             
    return True 

def get_vectorc_brute(index, matrixA):
    """Compute vector c using brute force method (dot products of matrixA columns)."""
    return [np.dot(matrixA[:, j], matrixA[:, index]) for j in range(matrixA.shape[1])]

def get_vectorc_model(index, model, input_dataset):
    """Compute vector c using model outputs."""
    input2 = model(input_dataset[index])  
    vectorc = []
    for j in range(len(input_dataset)):
        input1 = model(input_dataset[j])
        dot_product = torch.sum(input1 * input2, dim=1)
        vectorc.append(dot_product.detach().cpu().numpy().item())
    return vectorc
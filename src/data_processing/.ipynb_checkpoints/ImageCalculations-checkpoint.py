import torch
import numpy as np
from collections import defaultdict

import src.data_processing.BruteForceEstimator as bfEstimator
import src.data_processing.ImageCalculations as imgcalc
import src.data_processing.ImageProducts as ImageProducts


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

def get_unique_images(indices, intersection_indices, input_images, vectorb=None):
    """Get index of images that are translationally unique """
    
    images_union = [np.array(input_images[i]) for i in indices] 
    images_intersection = [np.array(input_images[j]) for j in intersection_indices]
    unique_indices = set(indices)
    similar_groups = defaultdict(set)
    for i, index_1 in enumerate(indices):
        for j, index_2 in enumerate(indices[i+1:]):  
            #-------------------------Use if dataset gets too large-----------------------
            #if round(vectorb[index_1], 7) == round(vectorb[index_2], 7):
            if not imgcalc.check_translationally_unique(images_union[i], images_union[i+1+j]):
                similar_groups[index_1].add(index_2)
                if index_1 in unique_indices:
                    unique_indices.remove(index_1)
                    
    unique_intersection_indices = set(intersection_indices)
    for i, index_1 in enumerate(intersection_indices):
        for j, index_2 in enumerate(intersection_indices[i+1:]):
            #-------------------------Use if dataset gets too large-----------------------
            #if round(vectorb[index_1], 7) == round(vectorb[index_2], 7):
            if not imgcalc.check_translationally_unique(images_intersection[i], images_intersection[i+1+j]):
                if index_1 in unique_intersection_indices:
                    unique_intersection_indices.remove(index_1)
    
    unique_indices_sorted = sorted(list(unique_indices))
    unique_intersection_indices_sorted = sorted(list(unique_intersection_indices)) 
    return similar_groups, unique_indices_sorted, unique_intersection_indices_sorted

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

def get_vectorb_model(index, model, input_dataset):
    """Compute vector b using model outputs."""
    vectorb = []
    for i in range(len(input_dataset)):
        scale = ImageProducts.scale_min(ImageProducts.ncc, -1)
        NCC_scaled_value = scale(input_dataset[index], input_dataset[i])
        vectorb.append(NCC_scaled_value)
        
    return vectorb
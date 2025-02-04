import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict

import src.data_processing.BruteForceEstimator as bfEstimator
import src.data_processing.ImageCalculations as imgcalc
import src.visualization.BFmethod as graphing
import src.visualization.Metrics as metrics
import src.visualization.ImagePlots as imgplt
import src.data_processing.ImageProducts as ImageProducts
import src.helpers.ModelUtilities as models

def plot_original_images(input1, input2, index1, index2):
    image1 = np.array(input1) 
    image2 = np.array(input2)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5)) 
    axes[0].imshow(image1, cmap='gray', interpolation='nearest')
    axes[0].set_title(f"Image {index1}")
    axes[0].axis("off")
    axes[1].imshow(image2, cmap='gray', interpolation='nearest')
    axes[1].set_title(f"Image {index2}")
    axes[1].axis("off")
    plt.tight_layout()
    plt.show()

def print_images(indices, intersection_indices, input_images):
    """Displays multiple images side by side, handling empty cases."""

    if len(intersection_indices) == 0:
        print("\nNo images in the intersection set to display.")
    else:
        print("\nPlotting images in the intersection set:")
        images_intersection = [np.array(input_images[j]) for j in intersection_indices] 
        
        if len(intersection_indices) == 1:
            fig, ax = plt.subplots(1, 1, figsize=(5, 5)) 
            ax.imshow(images_intersection[0], cmap='gray', interpolation='nearest')
            ax.set_title(f"Image {intersection_indices[0]}", fontsize=9)
            ax.axis("off")
        else:
            fig, axes = plt.subplots(1, len(intersection_indices), figsize=(10, 5))  
            for j, img in enumerate(images_intersection):
                axes[j].imshow(img, cmap='gray', interpolation='nearest')
                axes[j].set_title(f"Image {intersection_indices[j]}", fontsize=9)
                axes[j].axis("off")
                
        plt.tight_layout()
        plt.show()

    if len(indices) == 0:
        print("\nNo images in the union set to display.")
    else:
        print("\nPlotting images in the union set:")
        images_union = [np.array(input_images[i]) for i in indices] 

        if len(indices) == 1:
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            ax.imshow(images_union[0], cmap='gray', interpolation='nearest')
            ax.set_title(f"Image {indices[0]}", fontsize=9)
            ax.axis("off")
        else:
            fig, axes = plt.subplots(1, len(indices), figsize=(10, 5))  
            for j, img in enumerate(images_union):
                axes[j].imshow(img, cmap='gray', interpolation='nearest')
                axes[j].set_title(f"Image {indices[j]}", fontsize=9)  
                axes[j].axis("off") 

        plt.tight_layout()
        plt.show()

def plot_unique_images(vectorb, indices, intersection_indices, input_images):
    """Print images that are translationally unique """
    print(f"\nFor translationally unique plots: ")
    similar_groups, unique_indices, unique_intersection_indices = imgcalc.get_unique_images(indices, intersection_indices, input_images, vectorb)
    printing_translational_indices(similar_groups, unique_indices)
    print_images(unique_indices, unique_intersection_indices, input_images)

def printing_translational_indices(dictionary, indices):
    for i, index in enumerate(indices):
        target = indices[i]
        largest_group = set()

        for key, similar_set in dictionary.items():
            group = {key} | similar_set 
            if target in group and len(group) > len(largest_group):
                largest_group = group

        if largest_group:
            group_list = sorted(largest_group)
            group_list.remove(target)
            group_str = ", ".join(str(img) for img in group_list)
            print(f"Image {target} is translationally similar to Image {group_str}.")
        else:
            print(f"Image {target} has no translationally similar images.")
    
def display_and_plot_results(vectorb, vectorc, method_name, index, k, input_images):
    """Handle result display, table generation, and plotting."""
    kscore, indices, intersection_indices = metrics.get_k_neighbour_score(vectorb, vectorc, k)
    print(f"Estimating K-Score for Image {index}: K-Score = {kscore}")

    # print(f"Excluding translationally similar images")
    # kscore, indices, intersection_indices = metrics.get_k_neighbour_score_unique(vectorb, vectorc, k, input_images)
    # print(f"Estimating K-Score for Image {index}: K-Score = {kscore}")
    
    for vec, vec_name in [(vectorb, "b"), (vectorc, "c")]:
        top_values = sorted(enumerate(vec), key=lambda x: x[1], reverse=True)[:len(indices)]
        print(f"\nTop {len(indices)} values of Vector {vec_name}")
        for rank, (i, val) in enumerate(top_values, 1):
            print(f"Rank {rank}: Value = {val}, Index = {i}")
            
    indices = list(indices)   
    sorted_indices = sorted([i for i in indices if i != index])
    df = pd.DataFrame({
    "Index": [index] + sorted_indices, 
    "Vector b Value (NCC value)": [vectorb[index]] + [vectorb[i] for i in sorted_indices],
    "Vector c Value (Dot product value)": [vectorc[index]] + [vectorc[i] for i in sorted_indices]
    })
    
    print("\nComparison between vector c and vector b")
    print(df.to_string(index=False)) 
    
    print("\nComparing images in intersection & union sets")
    print_images(indices, intersection_indices, input_images)
    plot_unique_images(vectorb, indices, intersection_indices, input_images)
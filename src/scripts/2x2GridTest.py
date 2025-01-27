import matplotlib.pyplot as plt
from line_profiler import profile
import numpy as np
import src.data_processing.BruteForceEstimator as bfEstimator
import src.visualization.BFmethod as graphing
import src.visualization.Metrics as metrics

# -----Possible options-----
IMAGE_TYPES = ["NbinMmax_ones", "Nbin", "triangle", "triangle_mean_subtracted"]

IMAGE_FILTERS = ["unique", "Nmax_ones", "one_island"]

IMAGE_PRODUCT_TYPES = ["ncc", "ncc_scaled"]

EMBEDDING_TYPES = ["pencorr_D"]

# -----Variables-----
imageType = "2bin"
filters = ["100max_ones"]
imageProductType = "ncc_scaled_-1"
overwrite = {"imgSet": False, "imgProd": False, "embedding": False}
weight = None

k=2
pencorr_indices = list(range(3, 17))

bruteForceEstimator = bfEstimator.BruteForceEstimator(
    imageType=imageType, filters=filters, imageProductType=imageProductType, embeddingType="pencorr_5", overwrite=overwrite)
print("Image Set: ", bruteForceEstimator.imageSet)

# -----Execution-----

# Precompute the global min and max for the difference matrices
global_min = float('inf')
global_max = float('-inf')

# Compute the global min and max for the difference matrices
for i in pencorr_indices:
    embeddingType = f"pencorr_{i}"
    bruteForceEstimator = bfEstimator.BruteForceEstimator(
        imageType=imageType, filters=filters, imageProductType=imageProductType, embeddingType=embeddingType, overwrite=overwrite)

    matrixG = bruteForceEstimator.matrixG
    matrixA = bruteForceEstimator.matrixA

    if matrixG.ndim == 1:
        matrixG = matrixG.reshape(-1, 1)
    if matrixA.ndim == 1:
        matrixA = matrixA.reshape(-1, 1)
        
    dot_product_matrix = np.dot(matrixA.T, matrixA)
    diff_matrix = (matrixG - dot_product_matrix)**2
        
    global_min = min(global_min, diff_matrix.min())
    global_max = max(global_max, diff_matrix.max())

# Visualize
for i in pencorr_indices:
    embeddingType = f"pencorr_{i}"
    print(f"EmbeddingType: {embeddingType}")
    
    bruteForceEstimator = bfEstimator.BruteForceEstimator(
        imageType=imageType, filters=filters, imageProductType=imageProductType, embeddingType=embeddingType, overwrite=overwrite)

    matrixG = bruteForceEstimator.matrixG
    matrixA = bruteForceEstimator.matrixA
    # print("Matrix A: ", matrixA)
    # print("Matrix G: ", matrixG)

    if matrixG.ndim == 1:
        matrixG = matrixG.reshape(-1, 1)
    if matrixA.ndim == 1:
        matrixA = matrixA.reshape(-1, 1)
        
    # Approximation Method -- Squared Difference
    dot_product_matrix = np.dot(matrixA.T, matrixA)
    diff_matrix = (matrixG - dot_product_matrix)**2
    total_sum = np.sum(diff_matrix)
    print("Sum of all elements in difference matrix:", total_sum)
    
    # Appromixation Method -- KNN-IoU
    kscores=[]
    for i in range(len(matrixG)):
        vectorc=[]
        vectorb = bruteForceEstimator.matrixG[i]
        for j in range(len(matrixA[0])):
            vectorc.append(np.dot(bruteForceEstimator.matrixA[:, j], bruteForceEstimator.matrixA[:, i]))
        print(f"\nVector b of image {i}: {vectorb}")
        print(f"Vector c of image {i}: {vectorc}")
        kscore= metrics.get_k_neighbour_score(vectorb, vectorc, k)
        kscores.append(kscore)
        print(f"Estimating K-Score for Image {i}: K-Score = {kscore}")
        
    # Plot the K-scores
    plt.figure(figsize=(5, 3))
    plt.plot(range(1, len(kscores) + 1), kscores, marker='o', label=f"{embeddingType}")
    plt.xlabel("Image Index")
    plt.ylabel("K-score")
    plt.title(f"K-score Values for Each Image in {embeddingType}")
    plt.show()

    # Matrix G plot
    plt.figure(figsize=(18, 8))
    plt.subplot(1, 3, 1)
    plt.imshow(matrixG, cmap='viridis', aspect='auto')
    plt.title(f"Matrix G - {embeddingType}", fontsize=16)
    plt.colorbar()
    for (row, col), val in np.ndenumerate(matrixG):
        color = 'black'
        plt.text(col, row, f"{val:.2f}", ha='center', va='center', fontsize=8, color=color)

    # Dot Product Matrix plot
    plt.subplot(1, 3, 2)
    plt.imshow(dot_product_matrix, cmap='viridis', aspect='auto')
    plt.title(f"Dot Product of Matrix A - {embeddingType}", fontsize=16)
    plt.colorbar()
    for (row, col), val in np.ndenumerate(dot_product_matrix):
        color = 'black'
        plt.text(col, row, f"{val:.2f}", ha='center', va='center', fontsize=8, color=color)

    # Difference Matrix plot
    plt.subplot(1, 3, 3)
    plt.imshow(diff_matrix, cmap='pink', aspect='auto', vmin=global_min, vmax=global_max)
    plt.title(f"Difference (G - Dot Product) - {embeddingType}", fontsize=16)
    plt.colorbar()
    for (row, col), val in np.ndenumerate(diff_matrix):
        color = 'white' if val < 0 else 'black'
        plt.text(col, row, f"{val:.2f}", ha='center', va='center', fontsize=8, color=color)

    plt.tight_layout()
    plt.show()


import matplotlib.pyplot as plt
from line_profiler import profile
import numpy as np
import src.data_processing.BruteForceEstimator as bfEstimator
import src.visualization.BFmethod as graphing
import src.visualization.Metrics as metrics

# -----Possible options-----
IMAGE_TYPES = ["NbinMmax_ones", "Nbin", "triangles", "triangle_mean_subtracted"]

IMAGE_FILTERS = ["unique", "Nmax_ones", "one_island"]

IMAGE_PRODUCT_TYPES = ["ncc", "ncc_scaled"]

EMBEDDING_TYPES = ["pencorr_D"]

# -----Variables-----
imageType = "triangles"
filters = ["100max_ones"]
embeddingType = "pencorr_6"
imageProductType = "ncc_scaled_-1"
overwrite = {"imgSet": False, "imgProd": False, "embedding": False}
weight = None
k=2

bruteForceEstimator = bfEstimator.BruteForceEstimator(imageType=imageType, filters=filters, imageProductType=imageProductType,
                                                          embeddingType=embeddingType, overwrite=overwrite)
# -----Execution-----

# -----For Plotting of Image Sets-----
# Number of images in the dataset
# imageSet = bruteForceEstimator.imageSet
# max_images = 100
# numSample = min(len(imageSet), max_images)

# fig, axArr = plt.subplots(numSample, 1, figsize=(5, 5 * numSample))
# if numSample == 1:
#     axArr = [axArr]

# # Plot each image with a grid
# for imageIndex in range(numSample):
#     image = imageSet[imageIndex]
#     ax = axArr[imageIndex]
#     ax.imshow(image, cmap='Greys', interpolation='nearest')

#     # Highlight the grid lines
#     num_rows, num_cols = image.shape
#     for i in range(num_rows + 1):
#         ax.axhline(i - 0.5, color='red', linewidth=0.5)  
#     for j in range(num_cols + 1):
#         ax.axvline(j - 0.5, color='red', linewidth=0.5)  

#     ax.set_title(f"Image {imageIndex + 1}")
#     ax.axis('off')

# plt.subplots_adjust(hspace=0.5) 
# plt.show()

# -----For Calculations-----

# print(f"Image Set for {embeddingType}:\n", bruteForceEstimator.imageSet)
# print(f"Matrix G for {embeddingType}:\n", bruteForceEstimator.matrixG)

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
print("Sum of elements -- Squared difference matrix:", total_sum)

kscores=[]
for i in range(len(matrixG)):
    vectorc=[]
    vectorb = bruteForceEstimator.matrixG[i]
    for j in range(len(matrixA[0])):
        vectorc.append(np.dot(bruteForceEstimator.matrixA[:, j], bruteForceEstimator.matrixA[:, i]))
    # print(f"\nVector b of image {i}: {vectorb}")
    # print(f"Vector c of image {i}: {vectorc}")
    kscore= metrics.get_k_neighbour_score(vectorb, vectorc, k)
    kscores.append(kscore)
    #print(f"Estimating K-Score for Image {i}: K-Score = {kscore}")

# Plot the K-scores
plt.figure(figsize=(12, 3))
plt.plot(range(1, len(kscores) + 1), kscores, marker='o', label=f"{embeddingType}")
plt.xlabel("Image Index")
plt.ylabel("K-score")
plt.title(f"K-score Values for Each Image in {embeddingType}", pad=20)

top_10_indices = np.argsort(kscores)[-10:][::-1]
for idx in top_10_indices:
    plt.annotate(f"{idx+1}", (idx+1, kscores[idx]), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=8, rotation=45)
print("Top 10 K-Scores of Image Indexes: ",top_10_indices)
plt.show()

# Appromixation Method -- KNN-IoU

# for i in range(len(matrixG)):
#     vectorb = bruteForceEstimator.matrixG[i]
#     for i in range(len(matrixA)):
#     vectorc= bruteForceEstimator.matrixA

# -----For Plotting of Matrices-----
plt.figure(figsize=(24, 12))

# Matrix G plot
plt.subplot(1, 3, 1)
plt.imshow(matrixG, cmap='viridis', aspect='auto')
plt.title(f"Matrix G - {embeddingType}", fontsize=16)
plt.colorbar()
for (row, col), val in np.ndenumerate(matrixG):
    color = 'black'
    #plt.text(col, row, f"{val:.2f}", ha='center', va='center', fontsize=8, color=color)

# Matrix A plot
plt.subplot(1, 3, 2)
plt.imshow(matrixA, cmap='viridis', aspect='auto')
plt.title(f"Matrix A - {embeddingType}", fontsize=16)
plt.colorbar()
for (row, col), val in np.ndenumerate(matrixA):
    color = 'black'
    #plt.text(col, row, f"{val:.2f}", ha='center', va='center', fontsize=8, color=color)
# print(f"Matrix A for {embeddingType}:\n", bruteForceEstimator.matrixA)
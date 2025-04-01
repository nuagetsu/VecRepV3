import torch
import math
import numpy as np
from collections import defaultdict
from scipy.linalg import orthogonal_procrustes

import src.data_processing.ImageProducts as ImageProducts
import src.visualization.Metrics as metrics
import src.helpers.ModelUtilities as models
import src.helpers.FindingEmbUsingSample as LMethod

from src.data_processing import EmbeddingFunctions


def check_translationally_unique(img1: np.ndarray, img2: np.ndarray) -> bool:
    """Check if two binary images are translationally unique."""
    if img1.shape != img2.shape:
        return True  #shape different means confirm unique so they are unique

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
            if not check_translationally_unique(images_union[i], images_union[i+1+j]):
                similar_groups[index_1].add(index_2)
                if index_1 in unique_indices:
                    unique_indices.remove(index_1)
                    
    unique_intersection_indices = set(intersection_indices)
    for i, index_1 in enumerate(intersection_indices):
        for j, index_2 in enumerate(intersection_indices[i+1:]):
            #-------------------------Use if dataset gets too large-----------------------
            #if round(vectorb[index_1], 7) == round(vectorb[index_2], 7):
            if not check_translationally_unique(images_intersection[i], images_intersection[i+1+j]):
                if index_1 in unique_intersection_indices:
                    unique_intersection_indices.remove(index_1)
    
    unique_indices_sorted = sorted(list(unique_indices))
    unique_intersection_indices_sorted = sorted(list(unique_intersection_indices)) 
    return similar_groups, unique_indices_sorted, unique_intersection_indices_sorted

def find_bounding_box(image):
    rows, cols = np.where(image > 0)
    min_row, max_row = rows.min(), rows.max()
    min_col, max_col = cols.min(), cols.max()
    return min_row, max_row, min_col, max_col

def generate_all_translations(original_image):
    h, w = original_image.shape
    min_row, max_row, min_col, max_col = find_bounding_box(original_image)
    
    shape_h, shape_w = max_row - min_row + 1, max_col - min_col + 1
    
    # to ensure 3 pixel gap
    min_shift_x, max_shift_x = 3, w - shape_w - 3
    min_shift_y, max_shift_y = 3, h - shape_h - 3
    
    images = []
    
    for shift_y in range(min_shift_y, max_shift_y + 1):
        for shift_x in range(min_shift_x, max_shift_x + 1):
            new_image = np.zeros((h, w), dtype=int)
            
            for r in range(shape_h):
                for c in range(shape_w):
                    if original_image[min_row + r, min_col + c] > 0:
                        new_image[shift_y + r, shift_x + c] = 1
            
            images.append(new_image)
    
    return images

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

def get_vectorb(index, input_dataset):
    """Compute vector b using model outputs."""
    vectorb = []
    for i in range(len(input_dataset)):
        scale = ImageProducts.scale_min(ImageProducts.ncc, -1)
        NCC_scaled_value = scale(input_dataset[index], input_dataset[i])
        vectorb.append(NCC_scaled_value)
        
    return vectorb

def get_kscore_and_sets(vectorb, vectorc, k):
    kscore, indices, intersection_indices = metrics.get_k_neighbour_score(vectorb, vectorc, k)
    return kscore, indices, intersection_indices

def get_NCC_score(input1, input2):
    scale = ImageProducts.scale_min(ImageProducts.ncc, -1)
    NCC_scaled_value = scale(input1, input2)
    return NCC_scaled_value

def get_dp_score(input_vector1, input_vector2):
    dot_product_value = torch.sum(input_vector1 * input_vector2, dim=1) 
    return dot_product_value

def get_loss_value(dot_product_value, NCC_scaled_value):
    NCC_scaled_value = torch.tensor(NCC_scaled_value).to(dot_product_value.device).float()
    if NCC_scaled_value.ndim == 0:
        NCC_scaled_value = NCC_scaled_value.unsqueeze(0)

    loss_value = models.loss_fn_frobenius(dot_product_value, NCC_scaled_value)
    return loss_value.item()

def get_top_scores(vectorb, k, vectorc=None, print_results=True):
    top_values_b = []
    top_values_c = []

    rank = k

    top_values_b = sorted(enumerate(vectorb), key=lambda x: x[1], reverse=True)[:rank]
    
    if print_results:
        print(f"\nTop {rank} values of Vector b")
        for rank, (i, val) in enumerate(top_values_b, 1):
            print(f"Rank {rank}: Value = {val}, Index = {i}")

    if vectorc is not None:
        top_values_c = sorted(enumerate(vectorc), key=lambda x: x[1], reverse=True)[:rank]
        if print_results:
            print(f"\nTop {rank} values of Vector c")
            for rank, (i, val) in enumerate(top_values_c, 1):
                print(f"Rank {rank}: Value = {val}, Index = {i}")

    return top_values_b, top_values_c 


def get_bottom_scores(vectorb, k, vectorc=None, print_results=True):
    bottom_values_b = []
    bottom_values_c = []
    
    rank = k 

    bottom_values_b = sorted(enumerate(vectorb), key=lambda x: x[1])[:rank]
    
    if print_results:
        print(f"\nBottom {rank} values of Vector b")
        for rank, (i, val) in enumerate(bottom_values_b, 1):
            print(f"Rank {rank}: Value = {val}, Index = {i}")

    if vectorc is not None:
        bottom_values_c = sorted(enumerate(vectorc), key=lambda x: x[1])[:rank]
        if print_results:
            print(f"\nBottom {rank} values of Vector c")
            for rank, (i, val) in enumerate(bottom_values_c, 1):
                print(f"Rank {rank}: Value = {val}, Index = {i}")

    return bottom_values_b, bottom_values_c  


def kscore_loss_evaluation_model(imageset, input_dataset, model, k):
    kscores=[]
    losses=[]
    ncc_intervals = [round(i * 0.1, 1) for i in range(-10, 10)]  # [-1.0, -0.9, ..., 0.9]
    ncc_loss_dict = {
        f"{lower:.1f}-{lower + 0.1:.1f}": [] 
        for lower in ncc_intervals
    }

    epsilon = 1e-8
    for i in range(len(imageset)):
        vectorb = get_vectorb(i, imageset)
        vectorc = get_vectorc_model(i, model, input_dataset)
        kscore, _, _ = get_kscore_and_sets(vectorb, vectorc, k)
        kscores.append(kscore)
        #print(f"K-Score for index {i} is {kscore}")

        loss = []

        for j in range(len(imageset)):
            NCC_scaled_value =get_NCC_score(imageset[i], imageset[j])
            embedded_vector_image1 = model(input_dataset[i])
            embedded_vector_image2 = model(input_dataset[j])
            dot_product_value = get_dp_score(embedded_vector_image1, embedded_vector_image2)
            loss_value = get_loss_value(dot_product_value, NCC_scaled_value) 
            loss.append(loss_value)

            lower_bound = math.floor(NCC_scaled_value * 10) / 10

            # Handle edge case for values
            if NCC_scaled_value < -1.0:
                lower_bound = -1.0
            elif lower_bound > 0.9 or NCC_scaled_value == 1.0:
                lower_bound = 0.9
            interval_key = f"{lower_bound:.1f}-{lower_bound + 0.1:.1f}"

            ncc_loss_dict[interval_key].append(loss_value)

        average_loss = sum(loss) / len(loss)  

        #print(f"Average loss for index {i} is {average_loss}")
        losses.append(average_loss)
    
    return kscores, losses, ncc_loss_dict

def kscore_loss_evaluation_brute(imageset, matrixA, matrixG, k):
    kscores=[]
    losses=[]
    ncc_intervals = [round(i * 0.1, 1) for i in range(-10, 10)]  # [-1.0, -0.9, ..., 0.9]
    ncc_loss_dict = {
        f"{lower:.1f}-{lower + 0.1:.1f}": [] 
        for lower in ncc_intervals
    }

    epsilon = 1e-8
    for i in range(len(imageset)):
        vectorb = matrixG[i]
        vectorc = get_vectorc_brute(i, matrixA)
        kscore, _, _ = get_kscore_and_sets(vectorb, vectorc, k)
        kscores.append(kscore)
        #print(f"K-Score for index {i} is {kscore}")

        loss = []

        for j in range(len(imageset)):
            NCC_scaled_value =get_NCC_score(imageset[i], imageset[j])
            dot_product_value = np.dot(matrixA[:, i], matrixA[:, j])
            loss_value = get_loss_value(torch.tensor(dot_product_value), NCC_scaled_value) 
            loss.append(loss_value)

            lower_bound = math.floor(NCC_scaled_value * 10) / 10

            # Handle edge case for values
            if NCC_scaled_value < -1.0:
                lower_bound = -1.0
            elif lower_bound > 0.9 or NCC_scaled_value == 1.0:
                lower_bound = 0.9
            interval_key = f"{lower_bound:.1f}-{lower_bound + 0.1:.1f}"

            ncc_loss_dict[interval_key].append(loss_value)

        average_loss = sum(loss) / len(loss)  

        #print(f"Average loss for index {i} is {average_loss}")
        losses.append(average_loss)
    loss_per_ncc_score(ncc_loss_dict)
    return kscores, losses, ncc_loss_dict

def loss_per_ncc_score(ncc_loss_dict):
    average_loss_per_interval = {
        interval: sum(values)/len(values) if values else 0
        for interval, values in ncc_loss_dict.items()}

    print("\nNCC Interval\t\tAverage Loss")
    for interval in sorted(ncc_loss_dict.keys()):
        avg_loss = average_loss_per_interval[interval]
        count = len(ncc_loss_dict[interval])
        print(f"{interval}\t\t{avg_loss:.4f} ({count} samples)")
        
# def get_MSE(matrix1, matrix2):
#     difference_squared = (matrix1 - matrix2) ** 2
#     mean_squared_difference = np.sum(difference_squared) / difference_squared.size
#     return mean_squared_difference

def get_MSE(matrix1, matrix2):
    squared_diff = np.sum((matrix1 - matrix2) ** 2)
    total_energy = (np.sum(matrix1 ** 2) + np.sum(matrix2 ** 2)) / 2
    epsilon = 1e-8
    relative_msd = squared_diff / (total_energy + epsilon)
    return relative_msd

def get_vector_embeddings(input_dataset, model):
    num = len(input_dataset)
    model_vectors= []
    for i in range(num):
        embedded_vector_image = model(input_dataset[i])
        model_vectors.append(embedded_vector_image)
    return model_vectors

def get_embedding_estimate(matrixG, embeddingType, weight, index, imageSet):
    """
    :param image_input: Image of the same dimensions as that used in the training image set
    :param training_image_set: Image set used for training
    :param image_product: Image product used
    :param embedding_matrix: Embedding matrix for the image set
    :return: Estimated embedding for the input image
    """
    
    # imageProductVector = calculate_image_product_vector(image_input, training_image_set, get_image_product(image_product))
    imageProductVector = np.delete(get_vectorb(index, imageSet), index, axis=0)
    matrixG_new = np.delete(np.delete(matrixG, index, axis=0), index, axis=1)
    embedding_matrix = get_matrixA(matrixG_new, embeddingType, weight)
    
    estimateVector = LMethod.Lagrangian_Method2(embedding_matrix, imageProductVector)[0]
    return estimateVector

def get_vector_embeddings_LMethod(matrixG, embeddingType, weight, imageSet):
    num = len(imageSet)
    LMethod_vectors= []
    for i in range(num):
        embedded_vector_image = get_embedding_estimate(matrixG, embeddingType, weight, i, imageSet)
        LMethod_vectors.append(embedded_vector_image)
    return LMethod_vectors

def get_matrix_embeddings(input_dataset, model_vectors):
    num = len(input_dataset)
    model_matrix = torch.zeros(num, num)
    for i in range(num):
        for j in range (num):
            dot_product_value = torch.sum(model_vectors[i] * model_vectors[j], dim=1) 
            model_matrix[i, j] = dot_product_value
    return model_matrix

# Compute the matrix solution of the orthogonal (or unitary) Procrustes problem
def get_orthogonal_transformation(model_vectors, matrix):
    '''Given matrices A and B of the same shape, find an orthogonal matrix R that most closely maps A to B using this algorithm.
        matrix A in this case would be the embedding matrix obtained from the model, matrix B would be the Pencorr matrix (matrixA)'''
    embedding_model = torch.stack(model_vectors, dim=1)
    U, _ = orthogonal_procrustes(embedding_model.squeeze().detach().cpu().numpy().T, matrix)
    model_transformed = embedding_model.squeeze().detach().cpu().numpy().T @ U
    error_model = np.linalg.norm(model_transformed - matrix, 'fro')
    return model_transformed.T, error_model


def get_matrixG(imageSet, imageProductType):
    imageProduct = ImageProducts.get_image_product(imageProductType)
    imageProductMatrix = ImageProducts.calculate_image_product_matrix(imageSet, imageProduct)
    return imageProductMatrix

def get_matrixA(imageProductMatrix, embeddingType, weight):
    embeddingMatrix = EmbeddingFunctions.get_embedding_matrix(imageProductMatrix, embeddingType,
                                                                  weightMatrix=weight)
    return embeddingMatrix
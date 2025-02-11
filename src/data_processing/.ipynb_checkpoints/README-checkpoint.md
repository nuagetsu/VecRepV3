# data processing
The following will exemplify how we process our data to obtain our desired values.

# Overview
The methods described in the main readme are implemented here.

Each method, the brute force and sampling method, are made into their own object (BruteForceEstimator and SampleEstimator respectively)

Shape matching tests are also made into its own object.

Since the Sample method needs a test set in order to get meaningful results, the object SampleTester is needed.

BruteForceEstimator and SampleTester have inheritance from TestableEstimator, which is done so the same visualization functions can be used on both later.

## BruteForceEstimator
After the initialization function is called, it carries out the following 
1. Generate images set using ImageGenerators based on the imageType
2. Filters the images set using Filters based on the filters
3. Uses the filtered image set to generate the image product matrix (matrix G)
4. Applies the embedding method based on embeddingType to find the nearest correlation matrix (matrix G') and the embedding matrix (matrix A)
## SampleEstimator
After the initialization function is called, it carries out the following
1. Uses the input image set to generate an image product matrix (TAKE NOTE: This is NOT the matrix G used in visualization, as this is the training set. The data from the test set should be the one used in visualization)
2. Applies the embedding method based on embeddingType to find the nearest correlation matrix (matrix G') and the embedding matrix (matrix A)
## ShapeMatchingTester
After the initialization function is called, it carries out the following
1. Generates training images using ImageGenerators. Embeddings for these images are generated using the BF method.
2. Generates test images similarly. Embeddings for these images are generated using the Lagrangian method
3. Combines these two sets to create the full matching image set.
4. Creates a dictionary with an index/entry for each image. The key for each image is its index of the image
    in the full matching set. Each entry is another dictionary containing an "image" and an "embedding".
5. The tester also has a method to look for closest images to a particular shape. It takes images as inputs
    First, the image product scores between the input image and all other images is calculated
    When the input is within the matching set, it then looks for the corresponding entry in the dictionary and retrieves
        the corresponding embedding.
    When the input is not within the matching set, it generates a new embedding using the Lagrangian method.


SampleEstimators have a function get_embedding_estimate() which takes in an input image (of the same dimensions as images in the training set) and uses the sample images to calculate an image embedding for the input image

## SampleTester
After the initialization function is called, it carries out the following
1. Uses the test image set to generate the image product matrix (matrix G)
2. Uses the input sampleEstimator to generate vector embeddings for each of the test images. This generates the embedding matrix (matrix A)
3. Matrix G' is obtained by A.T A

## ImageProducts
There are many different types of Image products such as the normalised cross correlation [(NCC)](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/ell2.12516#ell212516-bib-0003), the mutual information [(MI)](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/ell2.12516#ell212516-bib-0004) and the structural similarity [(SSIM)](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/ell2.12516#ell212516-bib-0005) that we can employ to quantify similarities between different images. So far only NCC has been implemented under [ImageProducts.py](ImageProducts.py). 

Using the code we are able to obtain our image product matrix, which we will denote as $G$, which represents the image products between one randomly chosen image and every other image in a subset (The chosen image itself does not have to be included in the subset). 
We can also obtain an image product vector, known as $b$, which is essentially just the row in G which represents our chosen vector. 

![Matrix and vector representations](../../assets/Matrix_and_vector.png)

Monotonic transformations of the image product are also carried out here. The following monotonic transformations have
been implemented.
1. "scaled", e.g. "ncc_scaled_0.5"
    Scales the image product to fit a range of n to 1, e.g. 0.5 to 1
2. "pow", e.g. "ncc_pow_2"
    Image product to the power of n, e.g. power of 2 (squared)
3. "exp"
    e to the product of (image product - 1)
4. "log", e.g. "ncc_log_2"
    Log base n of image (n + image product - 1)
5. "base", e.g. "ncc_base_10"
    n raised to the power of (image product - 1)
6. "mult", e.g. "ncc_mult_0.5"
    Image product multiplied by scalar multiple

## Sampling Method
Allows us to sample a population of images. 
(i.e. take a small subset of matrices from the total set of matrices)

For example the total population of 3x3 binary images is 64, while the total population of 4x4 binary images is 4000+. This sampling method allows us to take a small sample of around 50 images from the 4000+ population of images.

Advantages include not having to produce all 900+ images which takes a significant amount of time, and that we are able to obtain a function that can have a vector representation of an image without having the image itself.

In testing, the main difference between the "Sampling Method" and BF Method is the set on which embeddings are tested on.

For the BF method, k-score is calculated on images within the training set.

For the Sampling Method, a separate test set is used and embeddings are calculated using the Lagrangian method before testing. I.e., Sampling Method tests the Lagrangian Method.

## EmbeddingFunctions

Contains methods used to generate embeddings as well as to find the nearest Positive Semidefinite matrix (mainly the latter).

The main method of note here is the Pencorr method and its corresponding Pencorr_Python method. The Pencorr_Python method is a translation of
the Matlab code used in Pencorr into Python for purposes of running on different devices (more details in matlab_functions).

The weighted version of Pencorr is also of note as this is required to use weighting matrices.

## ImageGenerators

Contains methods used to generate image sets. The following image sets are implemented:
1. "Nbin"
    Set of all NxN binary images.
2. "NbinMmax_ones"
    Set of all NxN binary images but with M% maximum elements occupied with 1s.
3. "triangles"
    Set of all 4x4 binary triangles within an 8x8 matrix. Examples are as follows.
    ![Examples of Triangles](../../assets/Triangle_examples.png)
4. "quadrilaterals"
    Set of all 4x4 binary quadrilaterals within an 8x8 matrix.
5. "shapes_s1_s2_dims_l_b"
    Set of all shapes of side lengths s1, s2,... with size of lxl and border size b. Example of 4 sided 5x5 shape in 9x9 matrix:
    ![Example of Quadrilateral](../../assets/Quad_example.png)
6. "randomshapes_s1_s2_dims_l_b_n"
    Random set of n shapes of side lengths s1, s2,... with size of lxl and border size b.
7. "NislandMmax_onesXimages"
    Set of X random images of size N, with M max ones. Imported from SamplingMethod file for organisation.

## Filters
A filter is a specific restriction we place on our sample of images, filters available are shown below:
1. Unique
Every matrix in the sample subset will be unique from each other i.e. no two matrices will be the same
2. Max Ones
A percentage is input into Max Ones. 
When creating binary matrices, the percentage of 1s will not exceed Max Ones. For example, 

Max Ones = 50%
An example binary 3x3 grid is 

```
[[0,0,0]
[0,1,1]
[1,1,0]]
```
The percentage of ones in the matrix is 44.4%
A second grid will be 

```
[[1,0,0]
[0,1,1]
[1,1,0]]
```
The percentages of ones in the second matrix is 55.6%
Therefore the first matrix will be accepted while the second matrix will be rejected under the Max Ones condition.

3. One Island
An island is defined as an isolated region of 1's (the 1's are connected to each other in the 4 cardinal directions)

For example in the matrix
```
[[1,1,1]
[0,1,0]
[0,0,1]]
```
The first island are the four continuously connected 1s that form the T-block, while the second island is the sole 1 in the bottom right corner of the matrix, meaning that this matrix has 2 islands.

The 1 island filter will only allow our sample of matrices to have 1 island in it.

All the three filters listed above can be used in conjunction, so that we can create different combinations of samples that are filtered differently, allowing us to analyse the different properties of said samples.

Here’s an updated README section that explains what each function does in your `imageCalculations.py` file:

---

## ImageCalculations

This module contains various methods used for evaluating models, particularly for image comparison and transformation. The functions help with tasks such as checking image uniqueness, computing similarity scores, evaluating losses, and performing transformations.

### **Function Descriptions**

#### **1. Image Uniqueness and Similarity Checks**
- **`check_translationally_unique(img1, img2) -> bool`**  
  Checks whether two binary images are translationally unique (i.e., they do not match under shifts). Returns `True` if they are unique and `False` if they are not.

- **`get_unique_images(indices, intersection_indices, input_images, vectorb=None)`**  
  Identifies and returns a list of images that are translationally unique. Also groups similar images together.

#### **2. Vector Calculations**
- **`get_vectorc_brute(index, matrixA)`**  
  Computes vector `c` using brute force by calculating dot products of matrix columns.

- **`get_vectorc_model(index, model, input_dataset)`**  
  Computes vector `c` using model outputs by taking dot products between model-generated embeddings.

- **`get_vectorb_model(index, model, input_dataset)`**  
  Computes vector `b` based on the normalized cross-correlation (NCC) similarity between images.

#### **3. Similarity Scores and Evaluation**
- **`get_kscore_and_sets(vectorb, vectorc, k)`**  
  Computes the k-nearest neighbor score using provided vectors and returns the score along with relevant indices.

- **`get_NCC_score(input1, input2)`**  
  Computes the normalized cross-correlation (NCC) score between two input images.

- **`get_dp_score(input_vector1, input_vector2)`**  
  Computes the dot product similarity score between two image embeddings.

- **`get_loss_value(dot_product_value, NCC_scaled_value)`**  
  Computes the loss value between the dot product and NCC-scaled values using the Frobenius norm.

#### **4. Model Evaluation and Loss Calculation**
- **`kscore_loss_evaluation(imageset, input_dataset, model, k)`**  
  Evaluates model performance using k-score and computes loss values based on image similarity.

- **`loss_per_ncc_score(ncc_loss_dict)`**  
  Computes and prints the average loss per NCC similarity score interval.

#### **5. Error Measurement**
- **`get_MSE(matrix1, matrix2)`**  
  Computes the mean squared error (MSE) between two matrices.

#### **6. Embeddings and Transformations**
- **`get_vector_embeddings(input_dataset, model)`**  
  Generates vector embeddings for images using a given model.

- **`get_matrix_embeddings(input_dataset, model_vectors)`**  
  Computes a similarity matrix from model embeddings using dot products.

- **`get_orthogonal_transformation(model_vectors, matrix)`**  
  Computes an orthogonal transformation to align the model's embedding matrix to a reference matrix using the orthogonal Procrustes problem.

---






















import src.visualization.ShapeMatching as graphing

SHAPE_IMAGE_TYPES = ["triangles", "quadrilaterals", "shapes_A_B_dims_C_D", "randomshapes_A_B_dims_C_D_E"]

"""
triangles: Set of all triangles within a 4 x 4 matrix with border of 2.

quadrilaterals: Set of all quadrilaterals within a 5 x 5 matrix with border of 2

shapes_A_B_dims_C_D: All shapes of sides A and B. No limit on number of shape sides to include. Dimensions of C and D.
C is the size of the shapes to be fit in (matrix of size C X C) and D is the size of the border.

randomshapes_A_B_dims_C_D_E: A reduced number of shapes of sides A and B. No limit on number of shape sides to include. 
Dimensions of C and D. C is the size of the shapes to be fit in (matrix of size C X C) and D is the size of the border. 
E is the number of shapes to generate (or size of the image set).
"""

IMAGE_FILTERS = ["unique", "Nmax_ones", "one_island"]

"""
one_island: Outputs a set of images such that each image has only one connected island of 1s 
(Diagonals are not connected)

Pmax_ones: Outputs a set of images such that each image has only P or lower percentage of 1s 
Sample input: 60max_ones

unique: Outputs a set of images such that each image is NOT a simple translation of another image 
in the set
"""

EMBEDDING_TYPES = ["pencorr_D", "dblcorr_D"]

"""
pencorr_D: Find the nearest correlation matrix using pencorr, subject to the rank constraint.
Then computes embeddings with D dimensions, then normalize the embeddings before output
Sample input: pencorr_200

dblcorr_D: Eigenvalue correction followed by Pencorr. Will have same performance as pencorr at lower rank constraint
but higher performance at higher rank constraint. Takes longer then pencorr.
Sample input: dblcorr_200
"""

# -----Variables-----
imageProductType = "ncc_pow_2"
embeddingType = "pencorr_500"
overwrite = {"imgSet": False, "imgProd": False, "embedding": False}
weight = "ncc_factor_1"
training_image_type = "randomshapes_3_4_dims_4_2_500"
training_filters = ["unique"]
test_image_type = "shapes_3_4_dims_4_2"
test_filters = ["unique"]

# -----Tests-----

# Default shape matching
"""
graphing.match_random_shape(training_image_type=training_image_type, training_filters=training_filters,
                            test_image_type=test_image_type, test_filters=test_filters,
                            embeddingType=embeddingType, weight=weight, imageProductType=imageProductType,
                            overwrite=overwrite)
"""

# Continuous shape matching - key in index into console

graphing.match_shapes_with_index(training_image_type=training_image_type, training_filters=training_filters,
                                 test_image_type=test_image_type, test_filters=test_filters,
                                 embeddingType=embeddingType, weight=weight, imageProductType=imageProductType,
                                 overwrite=overwrite)

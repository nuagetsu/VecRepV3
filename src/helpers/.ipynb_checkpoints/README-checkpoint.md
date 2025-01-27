## Generate Triangle
Note: This section was left by the previous interns and I have no idea why it is here. I did not find any code
relating to the generation of triangles in this section.

We want to generate a triangle filled with 1s.
We loop through the matrix, and use the algorithm below to figure out if the point is inside the triangle. If it is, it becomes a 1.

### Algorithm
![Algorithm to generate triangle](../../../master/assets/Generate_Triangle.png)

In order to find the location of P, we first find the areas of $\triangle PXY$, $\triangle PYZ$, and $\triangle PZX$, and if 

$$ \triangle PXY + \triangle PYZ + \triangle PZY = \triangle XYZ $$

we can say that the point P in located within the $\triangle XYZ$

In the second triangle, 

$$ \triangle QXY + \triangle QYZ + \triangle QZY > \triangle XYZ $$

therefore the point Q is not located within the $\triangle XYZ$, and the point will remain a zero.

## FilepathUtils

Utility methods used to generate filepaths. Filepaths are then used to save data sets or images which is important due to
exceedingly long computing times for each of these methods.

While all the methods here should be working, many are subject to the user's organisation preferences.

Data saved in the data folder should be backed up and moved or deleted if any of the methods here are changed.

## FindingEmbUsingSample

Contains methods related to the Langrangian Method of generating embeddings for new images not in the initial image set.

Two versions of the Lagrangian method exist here: One of them written by past interns that utilizes sympy while the new 
version only uses scipy and numpy methods. The latter of the two runs much faster and is now the main method used but was
written relatively late into development and has not undergone very extensive testing. As such, the previous method is also
left in as a backup.

Notably, the old Lagrangian Method function is the only one that uses the sympy package in this project.

Unit testing is required for the new Lagrangian Method.

## IslandCreator and NumIslands

Helper methods used to generate images.

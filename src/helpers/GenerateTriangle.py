import numpy as np

# define the three points as (x, y) tuples
p1 = (1, 1)
p2 = (5, 7)
p3 = (8, 4)
# create a n by n array of zeros
n = 10
def generate_triangle(p1, p2, p3, n):
    arr = np.zeros((n, n), dtype=int)

    # define a function to check if a point is inside the triangle
    def is_inside(p, p1, p2, p3):
        # use barycentric coordinates to determine if the point is inside the triangle
        # https://en.wikipedia.org/wiki/Barycentric_coordinate_system
        x, y = p
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        # calculate the area of the triangle
        area = abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2)
        # calculate the areas of the sub-triangles formed by the point and the vertices
        area1 = abs((x * (y2 - y3) + x2 * (y3 - y) + x3 * (y - y2)) / 2)
        area2 = abs((x1 * (y - y3) + x * (y3 - y1) + x3 * (y1 - y)) / 2)
        area3 = abs((x1 * (y2 - y) + x2 * (y - y1) + x * (y1 - y2)) / 2)
        # check if the sum of the sub-areas is equal to the area of the triangle
        return area == area1 + area2 + area3

    # loop through the array and fill the ones inside the triangle
    for i in range(n):
        for j in range(n):
            # check if the point (i, j) is inside the triangle
            if is_inside((i, j), p1, p2, p3):
                # set the value to one
                arr[i, j] = 1
    return arr


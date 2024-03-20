import random

import numpy as np


"""
Given an 3x3 dimensional grid,
[0,0,0]
[0,0,0]
[0,0,0]

Number of dots needed will be 4
An island is a sequence of connected dots etc.
[0,0,0]
[1,1,0]
[0,1,1]

Number of unique grids means grids must be unique lol
"""

def grid_fill(initial_coords, grid, curr_points):
    number_of_ones = 1
    possible_coords = set()
    if curr_points > len(grid[0]) ** 2:
        raise ValueError("Number of point to add to image too high")
    directions = [[-1,0], [1,0], [0,-1], [0,1]]
    while number_of_ones <= curr_points: #O(n**2)
        for direction in directions: #O(4)
            next_coords = (initial_coords[0] + direction[0], initial_coords[1] + direction[1])
            if 0 <= next_coords[0] < len(grid) and 0 <= next_coords[1] < len(grid[0]) and grid[next_coords[0]][next_coords[1]] != 1:
                possible_coords.add(next_coords)
        next_coords = random.choice(tuple(possible_coords)) #O(n**2????)
        possible_coords.remove(next_coords) #O(n**2)
        grid[next_coords[0]][next_coords[1]] = 1
        number_of_ones += 1
        initial_coords = next_coords
    return grid

def grid_creation(n, number_of_unique_grids, total_points):
    ret = []
    while len(ret) < number_of_unique_grids:
        grid = [[0 for _ in range(n)] for _ in range(n)]
        x = random.randint(0,n-1)
        y = random.randint(0,n-1)
        grid[x][y] = 1 #random initial point
        final_grid = grid_fill((x,y), grid, total_points)
        if final_grid in ret:
            continue
        else:
            ret.append(final_grid)
    return ret


if __name__ == "__main__":
    n = 10
    number_of_unique_grids = 300
    print(grid_creation(n, number_of_unique_grids))
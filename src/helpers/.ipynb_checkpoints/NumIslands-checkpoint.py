import numpy as np


def countIslands(grid) -> int:
    grid_copy = np.copy(grid)
    if len(grid_copy) == 0:
        return 0

    count = 0

    for i in range(len(grid_copy)):
        for j in range(len(grid_copy[0])):
            if grid_copy[i][j] == 1:
                mark_island(grid_copy, i, j, len(grid_copy), len(grid_copy[0]))
                count += 1

    return count


def mark_island(grid, i, j, m, n):
    if i < 0 or j < 0 or i >= m or j >= n or grid[i][j] != 1:
        return

    grid[i][j] = 2

    mark_island(grid, i, j + 1, m, n)
    mark_island(grid, i, j - 1, m, n)
    mark_island(grid, i + 1, j, m, n)
    mark_island(grid, i - 1, j, m, n)
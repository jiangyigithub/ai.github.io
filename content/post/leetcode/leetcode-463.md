---
title: 463. Island Perimeter
date: 2024-04-18 20:49:51+0800
description: Find the perimeter of the grid lands
tags: 
    - Matrix
    - DFS
categories:
    - LeetCode
math: true
---

Given a matrix, find the perimeter of the connected grid land 

# Intuition
Use DFS to find the island, then compute the perimeter.

# Approach
We use `result` to store the result and find all connected components of the island with DFS, to compute the perimeter,
to update `result`, we need to compute how many components that are connected the current component.

Now, note that to prevent infinite recursion, we set `grid[i][j] = -1` to mark it visited, then for each component, it may be in three states:
1. `grid[i][j] = 0`, it is water
2. `grid[i][j] = 1`, it is a component of the island and being unvisited
3. `grid[i][j] = -1`, it is a component of the island and has been visited.

# Complexity
- Time complexity: iterate all grids once.
$$O(mn)$$

- Space complexity: no extra space needed
$$O(1)$$

# Code
```c++
class Solution {
    vector<vector<int>> dirs{{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
public:
    void dfs(vector<vector<int>>& grid, int i, int j, int &result) {
        if (grid[i][j] != 1)    return;
        int num_edges = 4;
        grid[i][j] = -1;
        for (const auto&dir: dirs) {
            if (0 <= i + dir[0] && i + dir[0] < grid.size() && 
                0 <= j + dir[1] && j + dir[1] < grid[0].size()) {
                if (grid[i + dir[0]][j + dir[1]] == 0)      
                    continue;
                --num_edges;
                dfs(grid, i + dir[0], j + dir[1], result);
            }
        }
        result += num_edges;
    }

    int islandPerimeter(vector<vector<int>>& grid) {
        int result = 0;
        for (int i = 0; i < grid.size(); ++i) {
            for (int j = 0; j < grid[0].size(); ++j) {
                if (grid[i][j] == 0)    continue;
                dfs(grid, i, j, result);
            }
        }
        return result;
    }
};
```
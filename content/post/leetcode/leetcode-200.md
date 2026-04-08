---
title: 200. Number of Islands
date: 2024-04-19 20:25:03+0800
description: Count the number of Islands
tags: 
    - Matrix
    - DFS
categories:
    - LeetCode
math: true
---

Given a matrix where its grid component representing islands and waters, count the number of Islands.

# Intuition
Use DFS to find all connected components of the island, then count the number of islands.

# Approach
We iterate all grid component, when we meet the land, we use DFS to find all connected components of the island, and mark those connected components as visited.

# Complexity
- Time complexity: visit all positions once.
$$O(mn)$$

- Space complexity:
$$O(1)$$

# Code
```c++
class Solution {
    vector<vector<int>> dirs{{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
public:
    void dfs(vector<vector<char>>& grid, int i, int j) {
        if (grid[i][j] == '0')    return;
        grid[i][j] = '0';
        for (const auto &dir: dirs) {
            if (0 <= i + dir[0] && i + dir[0] < grid.size() &&
                0 <= j + dir[1] && j + dir[1] < grid[0].size()) {
                dfs(grid, i + dir[0], j + dir[1]);
            }
        }
    }

    int numIslands(vector<vector<char>>& grid) {
        int count = 0;
        for (int i = 0; i < grid.size(); ++i) {
            for (int j = 0; j < grid[0].size(); ++j) {
                if (grid[i][j] == '0')    continue;
                dfs(grid, i, j);
                ++count;
            }
        }
        return count;
    }
};
```
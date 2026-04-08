---
title: 1219. Path with Maximum Gold
date: 2024-05-14 20:39:03+0800
description: Find a path such that the sum is maximized.
tags: 
    - Matrix
    - Backtracking
    - Medium
categories:
    - LeetCode
math: true
---

Given a matrix, whose element representing the number of golds. Find a path such that the sum is maximized and without crossing the grids that has not gold elements.

# Intuition

Use backtracking to find all possible paths, and update the results.

# Approach

We use two variables `current` and `result` to store results, `current` stores the sum of golds from start to current position, `result` stores the final result. 

# Complexity

- Time complexity: In worst case, each grid contains gold, for each position, there are $\binom{m+n}{m}$ possible paths, so the overall complexity is

$$O\left(mn\binom{m+n}{m}\right)$$

- Space complexity: No extra spaces needed (without considering recursive stack)

$$O(1)$$

# Code

```c++
class Solution {
    vector<vector<int>> dirs{{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
public:
    void backtracking(vector<vector<int>>& grid, int i, int j, int &current, int &result) {
        if (grid[i][j] <= 0)    return;
        current += grid[i][j];
        // update result
        result = max(result, current);
        // mark as visited
        int temp = grid[i][j];
        grid[i][j] = -1;
        for (const auto &dir: dirs) {
            int row = i + dir[0], col = j + dir[1];
            if (row < 0 || row >= grid.size() || col < 0 || col >= grid[0].size()) {
                continue;
            }
            backtracking(grid, row, col, current, result);
        }
        // retrieve the state
        grid[i][j] = temp;
        current -= grid[i][j];
    }

    int getMaximumGold(vector<vector<int>>& grid) {
        int result = 0;
        for (int i = 0; i < grid.size(); ++i) {
            for (int j = 0; j < grid[0].size(); ++j) {
                if (grid[i][j] == 0)    continue;
                int current = 0;
                backtracking(grid, i, j, current, result);
            }
        }
        return result;
    }
};
```

# Reference

- [Leetcode 1219](https://leetcode.com/problems/path-with-maximum-gold/description/)

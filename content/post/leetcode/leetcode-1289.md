---
title: 1289. Minimum Falling Path Sum II
date: 2024-04-26 20:44:17+0800
description: Find the minimum falling path sum with no same column in adjacent rows
tags: 
    - Matrix
    - DP
    - Hard
categories:
    - LeetCode
math: true
---

Given a matrix, find the minimum falling path sum from top to bottom, with no two adjacent rows sharing the same column.

# Intuition

Same as the previous version, we only change the update formula.

# Approach

We use `dp[i][j]` to represent the minimum falling path sum that ends with `grid[i][j]`. The update rules is then given by

$$ dp[i][j] = \min_{k=1,\dots,n,k\neq j}(grid[i][j] + dp[i - 1][k]) $$

# Complexity

- Time complexity: We need to iterate all elements of the matrix once, and each iterate requires to iterate its last row once, which is $O(n)$. This can be reduced to $O(n^2\log n)$ by compute the smallest, second smallest elements of the last row.

$$O(n^3)$$

- Space complexity: the `dp` matrix is of size $n\times n$. This can be reduced to $O(n)$ by use a $n\times 2$ matrix since each row only relates to its last row.

$$O(n^2)$$

# Code

```c++
class Solution {
public:
    int minFallingPathSum(vector<vector<int>>& grid) {
        int n = grid.size();
        vector<vector<int>> dp(n, vector<int>(n, INT_MAX));
        dp[0] = grid[0];
        for (int row = 1; row < n; ++row) {
            for (int col = 0; col < n; ++col) {
                for (int last_col = 0; last_col < n; ++last_col) {
                    if (last_col == col)    continue;
                    dp[row][col] = min(dp[row][col], grid[row][col] + dp[row - 1][last_col]);
                }
            }
        }
        int result = INT_MAX;
        for (int i = 0; i < n; ++i) {
            result = min(result, dp[n - 1][i]);
        }
        return result;
    }
};
```

# Reference

- [Leetcode 1289](https://leetcode.com/problems/minimum-falling-path-sum-ii/description/)

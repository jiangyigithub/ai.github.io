---
title: 861. Score After Flipping Matrix
date: 2024-05-13 20:52:02+0800
description: Flip column or row such that the sum of the rows is maximized.
tags: 
    - Matrix
    - Greedy
    - Medium
categories:
    - LeetCode
math: true
---

Given a binary matrix, we can flip one column or one row, the goal is to flip zero or more times such that the sum of the number represented by the rows are maximized.

# Intuition

The leading `1`s are always important than the trailing `1`s. So we make sure that `1` appears before `0`s.

# Approach

The challenge is to determine when to flip the rows and flip the columns. From the intuition, we know that:

1. a row is flipped only if its first bit is `0`, after flipping, the number becomes larger and cannot be flipped again.
2. a column is flipped only if the number of `1`s are smaller than `0`s.

So we flip the rows first and the columns second.

# Complexity

- Time complexity: iterate matrix twice.
$$ O(mn) $$

- Space complexity: no extra space needed.
$$ O(1) $$

# Code

```c++
class Solution {
public:
    int matrixScore(vector<vector<int>>& grid) {
        int m = grid.size(), n = grid[0].size();
        for (int i = 0; i < m; ++i) {
            if (grid[i][0] == 0) {
                // flip row
                for (int j = 0; j < n; ++j) {
                    grid[i][j] = 1 - grid[i][j];
                }
            }
        }

        // check column by column
        for (int j = 1; j < n; ++j) {
            int count = 0;
            for (int i = 0; i < m; ++i) {
                count += grid[i][j];
            }
            if (count < (m + 1) / 2) {
                // flip column
                for (int k = 0; k < m; ++k) {
                    grid[k][j] = 1 - grid[k][j];
                }
            }
        }

        int result = 0;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (grid[i][j] == 0)    continue;
                result += (1 << (n - j - 1));
            }

        }

        return result;
    }
};
```

# References

- [Leetcode](https://leetcode.com/problems/score-after-flipping-matrix/description/)

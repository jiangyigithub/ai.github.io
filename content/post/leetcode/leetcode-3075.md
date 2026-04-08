---
title: 3075. Maximize Happiness of Selected Children
date: 2024-05-09 20:21:07+0800
description: Select some children such that the sum of their happiness is maximized
tags: 
    - Array
    - Sort
    - Medium
categories:
    - LeetCode
math: true
---

Given an integer array representing the happiness of the children, select $k$ children such that the sum of their happiness is maximized.

# Intuition

Since happiness of all rest children after choosing one child will decrease by 1, their relative order will still the same. So this problem is actually requiring us to select $k$ most happy children.

# Approach

Use sorting.

# Complexity

- Time complexity: Sort the array.
$$O(n\log n)$$

- Space complexity: No extra spaces are needed.
$$O(1)$$

# Code

```c++
class Solution {
public:
    long long maximumHappinessSum(vector<int>& happiness, int k) {
        sort(happiness.begin(), happiness.end());
        int n = happiness.size();
        long long result = 0;
        for (int i = 0; i < k; ++i) {
            result += max(happiness[n - 1 - i] - i, 0);
        }
        return result;
    }
};
```

# Reference

- [Leetcode](https://leetcode.com/problems/maximize-happiness-of-selected-children/description/)

---
title: 3068. Find the Maximum Sum of Node Values
date: 2024-05-09 20:21:07+0800
description: Perform XOR operations on some edges such that the sum is maximized
tags: 
    - Graph
    - Bit Manipulation
    - Hard
categories:
    - LeetCode
math: true
---

Given a graph and an integer $k$, where the nodes represent values, we can choose an edge, and perform XOR operations on its nodes corresponding to $k$. The goal is the find the maximum sum of the values after 0 or more XOR operations.

# Intuition

Since XOR operations satisfies the property that `a XOR b XOR b = a`, we can record the gain after XOR operation on an edge and finally obtain the result.

# Approach

We use `total_sum` to record the sum of the values of the original trees.

Then for each node, we perform the XOR operation and record the `change` if the `change > 0`, and this change requires one operation, which we add to `count`. Meanwhile, we use `positive_min` and `negative_max` to record the minimum absolute change for reverting use.

Now after all nodes are computed, we need to compute the result.

1. If `count` is even, it means the operations satisfied the requirement that the nodes of an edge changes simultaneously
2. If `count` is odd, then there is one invalid operation and we need to revert the operation. To make the final sum maximum, we can either subtract the `positive_min` or add `negative_max`, the result is then obtained by taking the maximum of them.

# Complexity

- Time complexity: iterate the array once.
$$O(n)$$

- Space complexity: No extra spaces are needed.
$$O(1)$$

# Code

```c++
class Solution {
public:
    long long maximumValueSum(vector<int>& nums, int k,
                              vector<vector<int>>& edges) {
        long long total_sum = 0;
        int count = 0;
        int positive_min = INT_MAX, negative_max = INT_MIN;
        for (int num : nums) {
            int new_num = num ^ k;
            total_sum += num;
            int change = new_num - num;
            if (change > 0) {
                positive_min = min(positive_min, change);
                total_sum += change;
                ++count;
            } else {
                negative_max = max(negative_max, change);
            }
        }

        if (count % 2 == 0)
            return total_sum;
        return max(total_sum - positive_min, total_sum + negative_max);
    }
};
```

# Reference

- [Leetcode](https://leetcode.com/problems/find-the-maximum-sum-of-node-values/description/)

---
title: 78. Subsets
date: 2024-05-21 22:16:49+0800
description: Find all subsets given an array
tags: 
    - Array
    - DFS
categories:
    - LeetCode
math: true
---

Given an array, find all subsets in the array

# Intuition

Use DFS to iterate over all subsets and record them.

# Approach

A subset can be represented as a binary number of `n` digits. Each digit is either `0` or `1`. We can use DFS to iterate over all possible such format of binary numbers.

# Complexity

- Time complexity: iterate all binary numbers of length `n` once.
$$O(2^n)$$

- Space complexity:
$$O(2^n)$$

# Code

```c++
class Solution {
public:
    void dfs(const vector<int>& nums, int index, vector<vector<int>>& result,
             vector<int>& s) {
        if (index == nums.size()) {
            result.push_back(s);
            return;
        }
        // digit is 1
        s.push_back(nums[index]);
        dfs(nums, index + 1, result, s);
        s.pop_back();
        // digit is 0
        dfs(nums, index + 1, result, s);
    }

    vector<vector<int>> subsets(vector<int>& nums) {
        vector<vector<int>> result;
        vector<int> s;
        dfs(nums, 0, result, s);
        return result;
    }
};
```

# Reference

- [Leetcode](https://leetcode.com/problems/subsets/description/)

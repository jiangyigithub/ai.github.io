---
title: 2997. Minimum Number of Operations to Make Array XOR Equal to K
date: 2024-04-29 21:16:53+0800
description: Minimum flip of bits of array elements to make the XOR of all elements equal to k
tags: 
    - Array
    - Bit manipulation
    - Medium
categories:
    - LeetCode
math: true
---

Given an integer array, we can flip one bit of one element of the array at every step, we asked to compute the *minimum* flips, such that the XOR of all elements of the array is equal to the given integer $k$.


# Intuition
XOR of multiple bits is always equal to either `1` or `0`, and changes one of the input bits will cause the result change to the other one.

# Approach
We first compute the XOR of all elements of the array, then we compare bit by bit with the XOR result and the given `k`, utils all bits becomes the same.

# Complexity
- Time complexity: compute the XOR result
$$O(n)$$ 

- Space complexity: No extra spaces are needed.
$$O(1)$$

# Code
```c++
class Solution {
public:
    int minOperations(vector<int>& nums, int k) {
        int result = nums[0];
        for (int i = 1; i < nums.size(); ++i) {
            result ^= nums[i];
        }
        vector<int> bits_result, bits_k;
        while (k) {
            bits_k.push_back(k % 2);
            k /= 2;
        }
        while (result) {
            bits_result.push_back(result % 2);
            result /= 2;
        }
        int count = 0;
        int i = 0;
        for (; i < bits_result.size() && i < bits_k.size(); ++i) {
            if (bits_result[i] != bits_k[i])
                ++count;
        }
        for (; i < bits_result.size(); ++i) {
            if (bits_result[i] == 1)
                ++count;
        }
        for (; i < bits_k.size(); ++i) {
            if (bits_k[i] == 1)
                ++count;
        }
        return count;
    }
};
```
---
title: 165. Compare Version Numbers
date: 2024-05-03 15:13:23+0800
description: Compare two version strings
tags: 
    - String
    - Two Pointer
    - Medium
categories:
    - LeetCode
math: true
---

Given two strings containing digits and dot, compare them in the form of a version number.


# Intuition
Since the leading zeros are required to be ignored, we use dot character as separator and compute the integer of each part and compare them individually.

# Approach
We use two index `i` and `j` to iterate over `s1` and `s2`, we move the index `i` to the next dot character and compute the integer `num1` we have found. Same operations for `j` to obtain the integer `num2`. Then `num1` and `num2` are compared.

# Complexity
- Time complexity: iterate each string once.
$$O(n)$$

- Space complexity: no extra space needed.
$$O(1)$$

# Code
```c++
class Solution {
public:
    int compareVersion(string version1, string version2) {
        int i = 0, j = 0;
        int m = version1.size(), n = version2.size();
        while (i < m || j < n) {
            int num1 = 0, num2 = 0;
            while (i < m && version1[i] != '.') {
                num1 = 10 * num1 + (version1[i++] - '0');
            }
            while (j < n && version2[j] != '.') {
                num2 = 10 * num2 + (version2[j++] - '0');
            }
            if (num1 < num2)    return -1;
            if (num1 > num2)    return 1;
            ++i, ++j;
        }
        return 0;
    }
};
```

# References
- [Leetcode](https://leetcode.com/problems/compare-version-numbers/description/)
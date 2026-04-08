---
title: 1137. N-th Tribonacci Number
date: 2024-04-24 18:53:26+0800
description: Sum of paths from root to leaf
tags: 
    - DP
categories:
    - LeetCode
    - Math
math: true
---

Compute the $n$-th tribonacci number.


# Intuition
Same as compute the $n$-th fibonacci number, we use three numbers to remember the state.

# Approach
We use three numbers to represent $n-2$, $n-1$ and $n$-th tribonacci number respectively

# Complexity
- Time complexity: 
$$O(n)$$ 

- Space complexity:
$$O(1)$$

# Code
```c++
class Solution {
public:
    int tribonacci(int n) {
        vector<int> nums{0, 1, 1};
        if (n < 3)  return nums[n];
        for (int i = 2; i < n; ++i) {
            int temp = nums[2];
            nums[2] += nums[0] + nums[1];
            nums[0] = nums[1];
            nums[1] = temp;
        }
        return nums[2];
    }
};
```
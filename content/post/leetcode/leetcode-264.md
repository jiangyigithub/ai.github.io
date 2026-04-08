---
title: 264. Ugly Number II
date: 2024-08-18 10:37:20+0000
description: Find the n-th ugly number
tags: 
    - DP
    - Medium
categories:
    - LeetCode
math: true
---

A number is `ugly` if its prime factors is a subset of $\{2, 3, 5\}$. We are required to find the $n$-th `ugly` number

# Intuition

Each ugly number is generated from a previous ugly number by multiplying $2$, $3$ or $5$.

# Approach

We use three pointers `index_2`, `index_3` and `index_5` to record the index of previous ugly number. For example, $4$ is generated from $2$ by multiplying $2$, so the index `index_2` is the index of `2`.

Then, we take the minimum of three generated numbers:

```c++
dp[i] = min(dp[index_2] * 2, dp[index_3] * 3, dp[index_5] * 5)
```

Finally, we need to update the pointers so that there is no duplication.

# Complexity

- Time complexity: iterate once.
$$O(n)$$

- Space complexity: Use a vector to store the result.
$$O(n)$$

# Code

```c++
class Solution {
public:
    int nthUglyNumber(int n) {
        vector<int> result{1};

        int index_2 = 0, index_3 = 0, index_5 = 0;

        for (int i = 1; i < n; ++i) {
            int result_2 = result[index_2] * 2;
            int result_3 = result[index_3] * 3;
            int result_5 = result[index_5] * 5;
            int current_ugly_num = min(result_2, min(result_3, result_5));

            if (current_ugly_num == result_2) ++index_2;
            if (current_ugly_num == result_3) ++index_3;
            if (current_ugly_num == result_5) ++index_5;
        }

        return result;
    }
};
```

# References

- [Leetcode 264](https://leetcode.com/problems/ugly-number-ii/description/)

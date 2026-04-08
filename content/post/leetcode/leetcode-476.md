---
title: 506. Relative Ranks
date: 2024-08-22 21:36:01+0800
description: Compute the complement of an integer
tags: 
    - Bit Manipulation
    - Easy
categories:
    - LeetCode
math: true
---

Given an integer, flip all bits in its binary representation.

# Intuition

Use XOR operation to complete this task.

# Approach

We can use the XOR operation to complete this task. However, notice that we cannot simply use `num ^ INT_MAX` since the leading `1`s are still `1`s if `num` is too small. Instead, we need to compute the number of digits in `num`, and use the corresponding masks.

Notice that if `2^30 < num <= 2^31-1`, in this case the corresponding mask will exceeds the integer ranges, thus we use `INT_MAX` directly in this case.

# Complexity

- Time complexity: Compute the digits of a number.
$$O(\log n)$$

- Space complexity: No extra spaces needed.
$$O(1)$$

# Code

```c++
class Solution {
public:
    int findComplement(int num) {
        int num_bits = 0;
        int temp = num;
        while (temp) {
            temp >>= 1;
            ++num_bits;
        }
        if (num_bits == 31) {
            return num ^ INT_MAX;
        }
        int mask = (1 << num_bits) - 1;
        return num ^ mask;
    }
};
```

# Reference

- [leetcode 476](https://leetcode.com/problems/number-complement/description)

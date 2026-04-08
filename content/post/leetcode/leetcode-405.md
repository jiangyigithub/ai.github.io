---
title: 405. Convert a Number to Hexadecimal
date: 2024-04-14 21:06:34+0800
description: Convert a decimal number to hexadecimal.
tags: 
    - Math
    - Bit Manipulation
categories:
    - LeetCode
math: true
---


# Intuition
Just use the transformation algorithm from decimal to hexadecimal

# Approach
Simulation by doing the following:
1. compute `remain = num % 16`, add `remain` to the result (`push_back`)
2. update `num = (num - remainder) / 16`
3. repeat step 1 and step 2 until `num` is 0

Notice that when `num < 0`, we need use its complement.


# Complexity
- Time complexity: the loop depends on the number of digits of `num`.
$$O(\log n)$$

- Space complexity:
$$ O(1) $$

# Code
```c++
class Solution {
public:
    string toHex(long num) {
        if (num == 0)   return "0";
        if (num < 0)    num = INT_MAX + num + 2 + INT_MAX;
        string result;
        while (num) {
            int divide = num % 16;
            num = (num - divide) / 16;
            if (divide < 10)    result += ('0' + divide);
            else result += ('a' + (divide - 10));
        }
        reverse(result.begin(), result.end());
        return result;
    }
};
```
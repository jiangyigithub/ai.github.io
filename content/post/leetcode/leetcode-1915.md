---
title: 1915. Number of Wonderful Substrings
date: 2024-05-02 10:51:29+0800
description: count the number of substrings that at most one character appears in an odd number of times.
tags: 
    - String
    - Bit Manipulation
    - Prefix Sum
    - Medium
categories:
    - LeetCode
math: true
---

Given a string ,count the number of wonderful substrings, where a **wonderful string** is a string that at most one character appears in an odd number of times.

# Intuition
We use the prefix sum and bitwise representation to solve the problem. The idea is that every string can be represented by a state, and we can count the number of wonderful substrings by performing operations among states


# Approach
According to the hint and the problem description, we have:
1. there are $10$ possible variables, which represent `a` to `j`
2. we only care about if the variable appears in odd number of times or in even number of times, which means that each variable has $2$ possible states.

so the number of overall possible states is $2^10=1024$. 
We define the prefix substring as `prefix[i] = s[1...i]`.

For each substring, we can represent the state of the prefix as an index of the array, that is, `state(prefix[i])` is a number between 0 and 1023, with each byte representing if the corresponding character appears in odd number of times (`1`) or in even number of times (`0`). A wonderful string is then defined as *a string whose state representation is 0 or a power of 2*.

now the substring `s[i...j]` is defined as `prefix[j]-prefix[i]`, in state representations, there are three cases:
1. if one character appears in odd number of times in both prefix substring, then in the result substring, the character appears in even number of times, which is equivalent to `(1, 1) -> 0`
2. if one character appears in even number of times in both prefix substring, then in the result substring, the character appears in even number of times, which is equivalent to `(0, 0) -> 0`
3. If one character appears in even number of times in one prefix substring, and appears in odd number of times in the other prefix substring, then in the result substring, the character appears in even number of times, which is equivalent to `(1, 0) -> 1`

so, according to analysis, the minus operation in string corresponding to the XOR operation in states. That is, the state of `s[i...j]` is given by `state(s[i...j])=state(prefix[j]) XOR state(prefix[i])`.

Note that there is also a simplification since if `a XOR b = c`, then `a XOR c = b`. Instead of using a for loop to compute all substrings that ended with character `j`:
```c++
for (int i = 0; i <j; ++i) {
    // substring s[i...j]
    int substr_state = state[j] ^ state[i];
    // check if substr is a wonderful substring
}
```
with the property of XOR, wo directly check if `state(prefix[j]) XOR 2^k` exists, that is:
```
for (int i = 0; i < 10; ++i) {
    int state = state[j] ^ (1 << i);
    // check if state exists and add them.
}
```

# Complexity
- Time complexity: iterate string once
$$O(n)$$

- Space complexity: stores the state representation.
$$O(1)$$

# Code
```c++
class Solution {
public:
    long long wonderfulSubstrings(string word) {
        vector<int> bits(1024);
        bits[0] = 1;
        long long result = 0;
        int prefix = 0;
        for (char c: word) {
            prefix ^= 1 << (c - 'a');
            result += bits[prefix];
            for (int i = 0; i < 10; ++i) {
                result += bits[prefix ^ (1 << i)];
            }
            ++bits[prefix];
        }
        return result;
    }
};
```


References:
- [Leetcode](https://leetcode.com/problems/number-of-wonderful-substrings/description/)
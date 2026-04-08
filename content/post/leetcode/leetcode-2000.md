---
title: 2000. Reverse Prefix of Word
date: 2024-05-01 09:01:22+0800
description: Given a string, revers its prefix ended with a specified character
tags: 
    - String
    - Easy
categories:
    - LeetCode
math: true
---

Given a string ans a specified character, reverse the prefix that is ended with the specified character.


# Intuition
Simulate the process.

# Approach
find the first occurrence of the specified character, then reverse the prefix.

# Complexity
- Time complexity: find the first occurrence of the specified character.
$$O(n)$$ 

- Space complexity: no extra spaces are needed.
$$O(1)$$

# Code
```c++
class Solution {
public:
    string reversePrefix(string word, char ch) {
        auto pos = word.find(ch);
        if (pos == word.npos)   return word;
        reverse(word.begin(), word.begin() + pos + 1);
        return word;
    }
};
```
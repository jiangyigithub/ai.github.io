---
title: 506. Relative Ranks
date: 2024-05-08 20:17:53+0800
description: Assign different Ranks to different scores
tags: 
    - Array
    - Sort
    - Easy
categories:
    - LeetCode
math: true
---

Given an array of scores, assign different ranks based on their position in the sorted array

# Intuition

Sort the array and assign based on the sorted array.

# Approach

in C++, we can use the property of the container `map` to solve this problem, the map is constructed so that the key is the score and the value is the index of the score.

# Complexity

- Time complexity: construct the map
$$O(n\log n)$$

- Space complexity: store the scores and indexes
$$O(n)$$

# Code

```c++
class Solution {
public:
    vector<string> findRelativeRanks(vector<int>& score) {
        map<int, int> m;
        for (int i = 0; i < score.size(); ++i) {
            m[score[i]] = i;
        }
        vector<string> result(score.size());
        int index = 1;
        for (auto iter = m.rbegin(); iter != m.rend(); ++iter, ++index) {
            if (index == 1) result[iter->second] = "Gold Medal";
            else if (index == 2)    result[iter->second] = "Silver Medal";
            else if (index == 3)    result[iter->second] = "Bronze Medal";
            else    result[iter->second] = to_string(index);
        }
        return result;
    }
};
```

# Reference

- [leetcode 506](https://leetcode.com/problems/relative-ranks/description)

---
title: 881. Boats to Save People
date: 2024-05-04 09:06:48+0800
description: Find the minimum number of boats to carry weighted people.
tags: 
    - Array
    - Two Pointer
    - Medium
categories:
    - LeetCode
math: true
---

Given an array `weight` where `weight[i]` representing the weight of person `i`, now given the capacity of the boat and the constraint that each boat can carry at most two people, find the minimum number of boats to carry all people.

# Intuition
Always pair the lightest person abd the heaviest person to a boat.

# Approach
We first sort the array `weight` in ascending order. Then we use two pointers `left=0` and `right=n-1` to iterate through the array. In each step, there are two cases:
1. If `weight[left]+weight[right] > limit`, then we cannot find a peer who can take one boat with `right`, in this case, `right` occupies a single boat alone.
2. If `weight[left]+weight[right] <= limit`, then these two people can take one boat.


# Complexity
- Time complexity: sort the array and iterate the array once.
$$ O(n\log n) $$

- Space complexity: no extra space needed.
$$ O(1) $$

# Code
```c++
class Solution {
public:
    int numRescueBoats(vector<int>& people, int limit) {
        sort(people.begin(), people.end());
        int left = 0, right = people.size() - 1;
        int num = 0;
        while (left <= right){
            if (people[right] + people[left] <= limit)
                ++left;
            --right;
            ++num;
        }
        
        return num;
    }
};
```


# References
- [Leetcode](https://leetcode.com/problems/boats-to-save-people/description/)
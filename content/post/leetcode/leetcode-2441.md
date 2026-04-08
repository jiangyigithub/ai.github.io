---
title: 2441. Largest Positive Integer That Exists With Its Negative
date: 2024-05-02 10:28:05+0800
description: Given a list of integers, find the largest positive integer that its negative also exists in the list.
tags: 
    - List
    - Hash Table
    - Sort
    - Two Pointer
    - Easy
categories:
    - LeetCode
math: true
---

Given a list containing integers, find the largest positive integer that its negative also exists in the list.


# Intuition
There are two ways to solve the problem. 
- One way is to use two sum, we find the all pairs of integers such that their sum is zero and the one is the negative of the other.
- The second way is to use two pointer, we move left and right pointer utils their absolute values are equal.

# Approach 1: two sum
Similar to Two Sum, for each `num` in `nums`, we store its negative `-num` in the hash table, 
however, notice that the added term can be determined by `num`, we can use a set instead.

## Complexity
- Time complexity: iterate the list once.
$$O(n)$$ 

- Space complexity: use a set to store numbers.
$$O(n)$$

## Code
```c++
class Solution {
public:
    int findMaxK(vector<int>& nums) {
        set<int> s;
        int result = -1;
        for (int num: nums) {
            if (s.find(-num) != s.end()) {
                result = max(result, abs(num));
            } else {
                s.insert(num);
            }
        }
        return result;
    }
};
```


# Approach 2: two pointer
We first sort the lists, then we use `left=0` and `right=length(nums)` pointer to iterate the list, there are three cases:
1. if `nums[left] == -nums[right]`, then we directly return `nums[right]` since this is the largest one (note that the list is sorted)
2. if `nums[left] < -nums[right]`, then we update `left` to `left + 1` since `(nums[left], -nums[left])` cannot be found in the list.
3. if `nums[left] > -nums[right]`, then we update `right` to `right - 1` since `(-nums[right], nums[right])` cannot be found in the list.

## Complexity
- Time complexity: sort the array and iterate the list once.
$$O(n\log n)$$ 

- Space complexity: no extra spaces are needed.
$$O(1)$$

## Code
```c++
class Solution {
public:
    int findMaxK(vector<int>& nums) {
        sort(nums.begin(), nums.end());
        int left = 0, right = nums.size() - 1;
        while (left <= right) {
            if (nums[left] == -nums[right]) return nums[right];
            else if (nums[left] < -nums[right]) ++left;
            else --right;
        }
        return -1;
    }
};
``

# References
- [Leetcode](https://leetcode.com/problems/largest-positive-integer-that-exists-with-its-negative/description/)
- [Leetcode Two Sum](https://leetcode.com/problems/two-sum/)
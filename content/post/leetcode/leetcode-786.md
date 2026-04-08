---
title: 786. K-th Smallest Prime Fraction
date: 2024-05-10 22:39:54+0800
description: k-th smallest prime fraction in an array
tags: 
    - Array
    - Priority Queue
categories:
    - LeetCode
math: true
---

Given an integer array of size $n$ containing prime integers, it can form $n(n-1)/2$ fractions, we are required to find the $k$-th smallest prime fraction.

# Intuition

We can use a priority queue to store prime integers, then we maintain the priority queue.

# Approach1: Brute force

## Complexity

- Time complexity: iterate the array once and maintain the priority queue.
$$O(n^2\log k)$$

- Space complexity: the size of the priority queue
$$O(k)$$

## Code

```c++
class Solution {
    class Compare {
    public:
        bool operator()(const vector<int>& a, const vector<int>& b){
            return 1.0 * a[0] / a[1] < 1.0 * b[0] / b[1]; // the root is the biggest
        }
    };
public:
    vector<int> kthSmallestPrimeFraction(vector<int>& arr, int k) {
        priority_queue<vector<int>, vector<vector<int>>, Compare> pq;
        int n = arr.size();
        for (int r = 1; r <= k + 1; ++r) {
            for (int i = 0; i < n; ++i) {
                if (i + r >= n) break;
                pq.push(vector<int>{arr[i], arr[i + r]});
                if (pq.size() > k) {
                    pq.pop();
                }
            }
        }
        return pq.top();
    }
};
```

# Approach2: Simplification

Notice that Approach1 requires iterating over all fractions, can we reduce the time complexity?

The solution is by considering the relative order, if we write a matrix whose element `a[i][j]=nums[i]/nums[j]` (`i<j`), then we know that `a[i][i+1]>...>a[n-1][n]` since the array `nums` are increasing. So, the smallest fraction are in `a[1][2], ..., a[n-1][n]`. If we take the smallest fraction, and add its successive elements (same column, last row), then we can find the second smallest fraction and so on. This solution requires iterating over $\max(n, k)$ fractions and $O(n)$ spaces.

## Complexity

- Time complexity: iterate $\max(n, k)$ elements  and maintain the priority queue.
$$O(\max(n, k)\log n)$$

- Space complexity: the size of the priority queue
$$O(n)$$

## Code

```c++
class Solution {
public:
    vector<int> kthSmallestPrimeFraction(vector<int>& A, int K) {
        // (fraction, (i, j))
        priority_queue<pair<double, pair<int, int>>> pq;
        for (int i = 0; i < A.size(); ++i) {
            pq.push({-1.0 * A[i] / A.back(), {i, A.size() - 1}});
        }
        while (--K) {
            auto t = pq.top().second;
            q.pop();
            --t.second;
            pq.push({-1.0 * A[t.first] / A[t.second], {t.first, t.second}});
        }
        return {A[pq.top().second.first], A[pq.top().second.second]};
    }
};
```

# Reference

- [leetcode](https://leetcode.com/problems/k-th-smallest-prime-fraction/description/)

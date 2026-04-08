---
title: 2370. Longest Ideal Subsequence
date: 2024-04-25 20:34:06+0800
description: Find the longest subsequence with given distance threshold on adjacent elements
tags: 
    - Hash Table
    - DP
    - Medium
categories:
    - LeetCode
math: true
---

Given a string consisting of lower-case characters, find the longest subsequence such that the distance between adjacent characters in the subsequence are less than a given threshold.


# Intuition
Same as the longest increasing subsequence, we can use dynamic programming to solve this problem.

# Approach
We use `dp[i]` to represent **the longest ideal subsequence that ended with `s[i - 1]`**. We then update the `dp[i]` with the property of ideal subsequence:
$$ dp[i] = \max_{j=1,\dots,i-1, \mathrm{abs}(s[i - 1]-s[j - 1])\leq k}(dp[i],\ dp[j] + 1) ,\ i=1,\dots,n$$

However, it turns out that the above solution is of complexity $O(n^2)$, which leads to *Time Exceed Limit*, so we need to optimize it.

Now, consider the property of ideal sequence, we only care about those characters that is within the range `(s[i - 1] - k, s[i - 1] + k)`. So, we can use a map `record`, whose key is all lowercase characters, to remember the result that is used to update `dp[i]`, that is, for given index `i`:

$$ record[l] = \max_{j=1,\dots,i - 1, s[j] - 'a' = l}dp[j]  $$
in this way, we have
$$ dp[i] = \max_{\mathrm{abs}((s[i]-'a') - l)\leq k}(dp[i],\ record[l] + 1),\ i=1,\dots,n $$
notice that `len(record)=26`, so the complexity now reduces to $O(n)$.

# Complexity
- Time complexity: iterate all characters once, each iteration queries `record` at most $26$ times. 
$$O(n)$$ 

- Space complexity: stores the `dp` array, can be optimized to $O(1)$ by update `result` at each iteration.
$$O(n)$$

# Code
```c++
class Solution {
public:
    int longestIdealString(string s, int k) {
        int n = s.size();
        vector<int> dp(n + 1);
        vector<int> record(26);
        for (int i = 0; i < n; ++i) {
            dp[i + 1] = 1;
            int index = s[i] - 'a';
            for (int j = 0; j < 26; ++j) {
                if (abs(index - j) <= k) {
                    dp[i + 1] = max(dp[i + 1], record[j] + 1);
                } 
            }
            record[index] = max(record[index], dp[i + 1]);
        }
        int result = 0;
        for (int i = 0; i < n + 1; ++i) {
            result = max(result, dp[i]);
        }
        return result;
    }
};
```
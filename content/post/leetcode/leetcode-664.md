---
title: 664. Strange Printer
date: 2024-08-21 15:16:41+0000
description: Use minimum turn to print a strange string.
tags: 
    - DP
    - Hard
categories:
    - LeetCode
math: true
---


# Approach

We can use a `dp` matrix, where element `dp[i][j]` represents the minimum turn to print substring `s[i...j]`.

Then, there are two cases:

- case 1: We print `s[i]` separately, now we have `dp[i][j] = 1 + dp[i + 1][j]`.
- case 2: There is a char `s[k] == s[i]`, then the string `s[i...k]` can be obtained by printing some new substrings on the range `s[i...k]`, now we have `dp[i][j] = dp[i][k-1]+dp[k+1][j]`

Combining case 1 and case 2, we can now write the update formula for dynamic programming:

```c++
dp[i][j] = 1 + dp[i + 1][j];
for (int k = i + 1; k <= j; ++k) {
    dp[i][j] = min(dp[i][j], dp[i][k-1] + dp[k+1][j]);
}
```

the base case is when `i > j`, `d[pi][j] = 0` and when `i=j`, `dp[i][j] = 1`.

# Complexity

- Time complexity: iterate the matrix once, each iteration takes $O(n)$ time.
$$O(n^3)$$

- Space complexity: use a `dp` matrix of size $n\times n$ to store the result
$$O(n^2)$$

# Code

```c++
class Solution {
public:
    int dfs(vector<vector<int>> &dp, int start, int end, const string &s) {
        if (start > end)    return 0;
        if (start == end)   return 1;
        if (dp[start][end] != -1)   return dp[start][end];

        int ans = 1 + dfs(dp, start + 1, end, s);
        for (int k = start + 1; k <= end; ++k) {
            if (s[k] != s[start])   continue;
            ans = min(ans, dfs(dp, start, k - 1, s) + dfs(dp, k + 1, end, s)); 
        }
        dp[start][end] = ans;
        return ans;
    }

    int strangePrinter(string s) {
        int n = s.size();
        vector<vector<int>> dp(n, vector<int>(n, -1));
        return dfs(dp, 0, n - 1, s);
    }
};
```

# References

- [Leetcode 664](https://leetcode.com/problems/strange-printer/description/)

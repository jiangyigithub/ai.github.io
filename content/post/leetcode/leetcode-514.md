---
title: 514. Freedom Trail
date: 2024-04-27 11:55:28+0800
description: Minimum operations to reconstruct a string from a ring string
tags: 
    - String
    - DP
    - DFS
    - Hard
categories:
    - LeetCode
math: true
---

Given a string displaying in the ring format, we can move one character at each step either in clockwise or anticlockwise (or hold still), now we need to retrieve a given string with minimum steps.


# Intuition
We can construct a graph from the string and use DFS to search all possible paths, and find the path with minimum steps.

# Approach
## DFS -> TLE
In the beginning, I am going to use DFS to find all possible paths and find the path with minimum steps. 

First, we need construct the graph, each character is adajcent to all other characters in `ring`, so there are `n=length(ring)` nodes and `(n-1)^n` edges. each edge has a weight representing the distance between two characters: `weight(i,j)=min(abs(i-j), n - abs(i-j))` (here we use index of the character to represent the node). 

Then, we can use DFS to search all possible paths, in each point, we have a state `(ring_index, key_index)`, representing the current index on `ring` and `key`, there are two cases:
1. `ring[ring_index] == key[key_index]`, in this case, we change `key_index` to `key_index+1` and keeps `ring_index` unchanged (hold still).
2. `ring[ring_index] != key[key_index]`, in this case, we need to rotate the string `ring` to make `ring[ring_index] == key[key_index]`, this takes step and notice that there may multiple choices, so we need to go over all of them.

Once `key_index == len(key)`, we have found a path and we can now update the result.

The code is given as follows:

```c++
class Solution {
public:
    void dfs(unordered_map<int, vector<vector<int>>> &adjs, 
             const string &ring, const string &key, 
             int ring_index, int key_index, 
             int &min_rotates, int &current_rotates) {
        if (key_index == key.size()) {
            min_rotates = min(min_rotates, current_rotates);
            return;
        } 

        // character matches
        if (ring[ring_index] == key[key_index]) {
            dfs(adjs, ring, key, ring_index, key_index + 1, min_rotates, current_rotates);
            return;
        }
        // character doesn't match
        for (const auto &next_node: adjs[ring_index]) {
            int next_ring_index = next_node[0], rotates = next_node[1];
            if (ring[next_ring_index] != key[key_index])    continue;
            current_rotates += rotates;
            dfs(adjs, ring, key, next_ring_index, key_index, min_rotates, current_rotates);
            current_rotates -= rotates;
        }
    }

    int findRotateSteps(string ring, string key) {
        unordered_map<int, vector<vector<int>>> adjs;
        int n = ring.size();
        for (int i = 0; i < ring.size(); ++i) {
            for (int j = 0; j < ring.size(); ++j) {
                if (j == i) continue;
                int diff = abs(i - j);
                int rotates = min(diff, n - diff);
                adjs[i].push_back(vector<int>{j, rotates});
            }
        }

        int ring_index = 0, key_index = 0;
        int min_rotates = INT_MAX, current_rotates = 0;
        dfs(adjs, ring, key, ring_index, key_index, min_rotates, current_rotates);
        int spell = key.size();
        return min_rotates + spell;
    }
};
```

## Dynamic programming
The problem of DFS is that, its time complexity grows exponentially if there have multiple repeat characters, which causes TLE (time limit exceeded) error.

So, to reduce the complexity, we can construct the solution from bottom to up. That is, we remember the path to go from the current state, and now we now go one step back, util we back to the original state.

We use `dp[i][j]` to represent start from `ring[j]`, the minimum rotate steps we need to recover the string `key[i...m]`, where `m=length(key)`, the target then becomes finding out `dp[0][0]`.

Note that we can easily compute `dp[m-1][j]`, `j=1,\dots,n`, since there is only one character we need to recover, so we start from `ring[j]`, and rotate util we find a character `ring[k]` such that `ring[k]==key[m-1]`, the minimum steps is then updated. The update formula is then given by

$$ dp[i][j] = \min_{k=1,\dots,n,\ ring[k]=key[i]}(dp[i][j],\ dp[i + 1][k] + step(j, k)) $$

where $step(j,k)=\min(|j-k|,\ n-|j-k|)$.

# Complexity
- Time complexity: The `dp` matrix is of size $m\times n$, and each time, we need to iterate over the `ring` once.
$$O(mn^2)$$ 

- Space complexity:The `dp` matrix is of size $m\times n$
$$O(mn)$$

# Code
```c++
class Solution {
public:
    int findRotateSteps(string ring, string key) {
        int m = key.size(), n = ring.size();
        vector<vector<int>> dp(m + 1, vector<int>(n));
        for (int i = m - 1; i >= 0; --i) {
            // start from ring[j]
            for (int j = 0; j < n; ++j) {
                dp[i][j] = INT_MAX;
                // Find the feasible target ring[k] == key[i]
                for (int k = 0; k < n; ++k) {
                    if (ring[k] != key[i])
                        continue;
                    int rotates = min(abs(k - j), n - abs(k - j));
                    dp[i][j] = min(dp[i][j], dp[i + 1][k] + rotates);
                }
            }
        }
        return dp[0][0] + m;
    }
};
```


# Reference
- [leetcode 514](https://leetcode.com/problems/freedom-trail/description)
- [Solution](https://www.cnblogs.com/grandyang/p/6675879.html)
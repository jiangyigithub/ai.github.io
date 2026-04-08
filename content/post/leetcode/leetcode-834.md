---
title: 834. Sum of Distances in Tree
date: 2024-04-29 21:24:06+0800
description: Sum os distance of all nodes to all other nodes in a tree
tags: 
    - Tree
    - BFS
    - Hard
categories:
    - LeetCode
math: true
---

Given a tree, return a vector, with each elements of the vector is the sum of the distances between the corresponding node and all other nodes.

# Intuition
Intuitively, I want to use DFS + memorization to solve the problem, however, such method exceeds te time limit. 

Then, I refer to some solutions, and solve the problem.

# Approach
## DP -> TLE
In the first, I am thinking that we can use `dp[i][j]` to represent the distance between the node `i` and the node `j`. Initially, `dp` are initialized as follows:
1. If there is an edge between `i` and `j`, then `dp[i][j]=dp[j][i]=1`.
2. If there is an edge between `i` and `j`, then `dp[i][j]=dp[j][i]=INFINITY`.

we traversal from node `0` to node `n-1`, for each node `i`, we compute the distance between `i` and all other nodes `j`. There are two cases:
1. `dp[i][j] != INFINITY`, then we directly returns `dp[i][j]`
2. `dp[i][j] == INFINITY`, then it means the distance between node `i` and node `j` hasn't been computed, we then update it as follows:

$$ dp[i][j] = \min_{k\in N(i)} (1 + dp[k][j]) $$

where $N(i)$ is the nodes that adjacent to node `i`. The `min` operation is used here since the node `k` and the node `j` may not connected (without passing node `i`).

The code is given as follows. However, the time complexity is $O(n^2)$, which exceeds the time limit.

```c++
class Solution {
public:
    int dfs(const vector<vector<int>>& adjs, vector<vector<int>> &dp, 
        vector<bool> &visited, int start, int end) {
        if (dp[start][end] != INT_MAX)    return dp[start][end];
        if (visited[start]) return INT_MAX;
        visited[start] = true;
        int distance = INT_MAX;
        for (int next_node: adjs[start]) {
            distance = min(distance, dfs(adjs, dp, visited, next_node, end));
        }
        visited[start] = false;
        if (distance != INT_MAX)    dp[start][end] = distance + 1;
        return dp[start][end];
    }

    vector<int> sumOfDistancesInTree(int n, vector<vector<int>>& edges) {
        vector<vector<int>> dp(n, vector<int>(n, INT_MAX));
        vector<vector<int>> adjs(n);
        // initialize the distance matrix
        for (const auto& edge: edges) {
            dp[edge[0]][edge[1]] = 1;
            dp[edge[1]][edge[0]] = 1;
            adjs[edge[0]].push_back(edge[1]);
            adjs[edge[1]].push_back(edge[0]);
        }

        vector<bool> visited(n, false);
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                dfs(adjs, dp, visited, i, j);
            }
            // the graph is undirected
            for (int j = 0; j < i; ++j) {
                dp[i][j] = dp[j][i];
            }
        }

        vector<int> result(n);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (j == i) continue;
                result[i] += dp[i][j];
            }
        }
        return result;
    }
};
```

## Tree + Traversal
The second way is not easy to figure out. We decompose it into two parts:

1. compute the distance between the root node and all other nodes.
2. Convert the root node from one to another and update the result.

### Sum of distance from root node to all other nodes.
Consider one example tree with root set as `0`:

```c++
  0
 /  \
1    2 
   / | \
  3  4  5
```

We first define `dist[i]` as the distance of all child nodes of node `i` to node `i`. Then we have:

1. $dist[i] = 0$ if `i` is a leaf node.
2. $dist[i] = \sum_{j\in C(i)}dist[j] + |C(i)|$ if `i` is not a leaf node, where $C(i)$ is the offspring of node `i`

We can now compute the sum of distances between root `0` to all other nodes, which is `result[0]`.

Now we need to compute all other results. Repeating the above process is too time consuming, we need to reduce the time complexity. We are seeking a way to compute `result[i]` from `result[j]`, where $j\in C(i)$. 

Note that for $k\in C(j)$, when we compute `result[i]`, we are computing distance from $k$ to $i$, so we need to reduce by 1 since their height decreases (the root changes from `i` to `j`). On the other hand, all other nodes, which are not offspring of node `j`, is added by 1 since the height of them increases. Thus, the transformation reads:

$$ result[j] = result[i] - |C(j)| + n - |C(j)| $$

# Complexity

- Time complexity: visited all nodes twice.
$$ O(n) $$

- Space complexity: use two vectors of size $n$ to store the result.
$$ O(n) $$

# Code
```c++
class Solution {
public:
    void post_order_traversal(const vector<vector<int>>& adjs,
                              vector<int>& count, vector<int>& result,
                              int node, int parent) {
        for (int next_node : adjs[node]) {
            if (next_node == parent)
                continue;
            post_order_traversal(adjs, count, result, next_node, node);
            count[node] += count[next_node];
            result[node] += count[next_node] + result[next_node];
        }
        ++count[node];
    }

    void pre_order_traversal(const vector<vector<int>>& adjs,
                             vector<int>& count, vector<int>& result,
                             int node, int parent) {
        for (int next_node : adjs[node]) {
            if (next_node == parent)
                continue;
            result[next_node] =
                result[node] - count[next_node] + count.size() - count[next_node];
            pre_order_traversal(adjs, count, result, next_node, node);
        }
    }

    vector<int> sumOfDistancesInTree(int n, vector<vector<int>>& edges) {
        vector<int> count(n), result(n);
        vector<vector<int>> adjs(n);
        for (const auto &edge: edges) {
            adjs[edge[0]].push_back(edge[1]);
            adjs[edge[1]].push_back(edge[0]);
        }
        // compute result[0]
        post_order_traversal(adjs, count, result, 0, -1);
        // compute other results from result[0]
        pre_order_traversal(adjs, count, result, 0, -1);

        return result;
    }
};
```


# References
- [Leetcode](https://leetcode.com/problems/sum-of-distances-in-tree/description/)
- [Solution](https://www.cnblogs.com/grandyang/p/11520804.html)
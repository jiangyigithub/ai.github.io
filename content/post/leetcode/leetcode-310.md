---
title: 310. Minimum Height Trees
date: 2024-04-23 21:02:07+0800
description: Construct a tree with a minimum height
tags: 
    - Tree
    - BFS
    - Topological sort
categories:
    - LeetCode
math: true
---

Given a tree, reorganize the tree such that the height of the tree is minimized.


# Intuition
We construct the tree from bottom to top, util we find the root of the tree

# Approach
We use topological sort to order the nodes of the tree, then we iteratively construct the tree from bottom to top with BFS.

We use a different stop criteria to avoid missing possible solutions.

> The result contains at most two possible roots, since if there are three, then the degree of one node must be lower than the other two nodes.

# Complexity
- Time complexity: iterate all nodes once.
$$O(n)$$ 

- Space complexity: store the adjacent lists.
$$O(n)$$

# Code
```c++
class Solution {
public:
    vector<int> findMinHeightTrees(int n, vector<vector<int>>& edges) {
        // boundary check
        if (n == 1) return vector<int>{0};
        vector<vector<int>> adjs(n);
        vector<int> in_degrees(n);
        for (const auto &edge: edges) {
            adjs[edge[0]].push_back(edge[1]);
            adjs[edge[1]].push_back(edge[0]);
            ++in_degrees[edge[0]];
            ++in_degrees[edge[1]];
        }
        // leaf nodes
        queue<int> q;
        for (int i = 0; i < n; ++i) {
            if (in_degrees[i] == 1) q.push(i);
        }
        while (n > 2) {
            // construct next layer
            int size = q.size();
            n -= size;
            for (int i = 0; i < size; ++i) {
                int node = q.front();
                for (int next_node: adjs[node]) {
                    if (--in_degrees[next_node] == 1) {
                        q.push(next_node);
                    }
                }
                q.pop();
            }
        }
        vector<int> result;
        while (!q.empty()) {
            result.push_back(q.front());
            q.pop();
        }
        return result;
    }
};
```
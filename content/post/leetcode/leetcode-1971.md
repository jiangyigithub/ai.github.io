---
title: 1971. Find if Path Exists in Graph
date: 2024-04-21 10:51:46+0800
description: Path from source to destination in a given graph
tags: 
    - Graph
    - BFS
categories:
    - LeetCode
math: true
---

Find a available path from a given source to a given destination in a given graph.

# Intuition
Use DFS to find all reachable nodes and check if the destination lie within those nodes.

# Approach
We first transform the adjacency matrix to adjacency list to make BFS easier, then we use 
a queue to maintain the reachable nodes, to prevent from cycling, we also use a set to keep track of visited nodes.
If at any point, we reach the `destination`, we return directly.

# Complexity
- Time complexity: iterate all nodes at most once
$$O(n)$$

- Space complexity: stores nodes of the same distance in the graph to the source
$$O(n)$$

# Code
```c++
class Solution {
public:
    bool validPath(int n, vector<vector<int>>& edges, int source, int destination) {
        unordered_map<int, vector<int>> adjs;
        for (const auto &edge: edges) {
            adjs[edge[0]].push_back(edge[1]);
            adjs[edge[1]].push_back(edge[0]);
        }

        unordered_set<int> visited;
        queue<int> q;
        q.push(source);
        while (!q.empty()) {
            int node = q.front();
            q.pop();
            if (node == destination)    return true;
            for (int next_node: adjs[node]) {
                if (visited.find(next_node) != visited.end())   
                    continue;
                q.push(next_node);
                visited.insert(next_node);
            }
        }
        return false;
    }
};
```
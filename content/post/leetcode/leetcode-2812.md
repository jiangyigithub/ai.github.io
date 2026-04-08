---
title: 2812. Find the Safest Path in a Grid
date: 2024-05-15 20:24:56+0800
description: Find the path with minimum weights.
tags: 
    - Matrix
    - Weighted Graph
    - DFS
    - Medium
categories:
    - LeetCode
math: true
---

Given a matrix, we wish to find a path that from start to end, such that the path is as far as from the dangerous position.

# Intuition

We can convert the problem into finding a path with minimum weights in a graph.

# Approach

We first convert the matrix into a graph, with node as position `(i, j)` and weight `w[i,j]\min_k(|i-thief[k][0]| + |j - thief[k][1]|)`, where `thief[k]=(thief[k][0], thief[k][1])` is the position of the thief `thief[k]`. To do this, we can use DFS, starting from each thief and iterate through the grids.

Then, we need to find a path from the start `(0, 0)` to target `(n - 1, n - 1)`. This can be done via Dijkstra's Algorithm. We use a priority queue to keep track of minimum to-be-visited nodes, this ensures that the newly added nodes are always with the maximum safe factor.

# Complexity

- Time complexity: iterate all elements twice, the second trial requires to maintain the priority queue.
$$O(n^2\log n)$$

- Space complexity: store safe factors matrix.
$$O(n^2)$$

# Code

```c++
class Solution {
    vector<vector<int>> dirs{{-1, 0}, {1, 0}, {0, -1}, {0, 1}};

public:
    int maximumSafenessFactor(vector<vector<int>>& grid) {
        int n = grid.size();
        if (grid[0][0] == 1 || grid[n - 1][n - 1] == 1)
            return 0;

        vector<vector<int>> scores(n, vector<int>(n, INT_MAX));
        queue<pair<int, int>> q;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (grid[i][j] == 0)
                    continue;
                q.push(make_pair(i, j));
                scores[i][j] = 0;
            }
        }
        while (!q.empty()) {
            const auto node = q.front();
            int x = node.first, y = node.second;
            for (const auto& dir : dirs) {
                int new_x = x + dir[0], new_y = y + dir[1];
                if (new_x < 0 || new_x >= n || new_y < 0 || new_y >= n)
                    continue;
                if (scores[new_x][new_y] <= 1 + scores[x][y])
                    continue;
                scores[new_x][new_y] = 1 + scores[x][y];
                q.push(make_pair(new_x, new_y));
            }
            q.pop();
        }

        vector<vector<bool>> visited(n, vector<bool>(n, false));
        priority_queue<pair<int, pair<int, int>>> pq;
        pq.push(make_pair(scores[0][0], make_pair(0, 0)));

        while (!pq.empty()) {
            auto node = pq.top();
            pq.pop();
            int safe_factor = node.first;
            auto pos = node.second;
            if (pos.first == n - 1 && pos.second == n - 1)
                return safe_factor;
            visited[pos.first][pos.second] = true;
            for (const auto& dir : dirs) {
                int new_x = pos.first + dir[0], new_y = pos.second + dir[1];
                if (new_x < 0 || new_x >= n || new_y < 0 || new_y >= n)
                    continue;

                if (visited[new_x][new_y])
                    continue;
                
                int score = min(safe_factor, scores[new_x][new_y]);
                pq.push(make_pair(score, make_pair(new_x, new_y)));
                visited[new_x][new_y] = true;
            }
            
        }

        return -1;
    }
};
```

# References

- [Leetcode](https://leetcode.com/problems/find-the-safest-path-in-a-grid/description)

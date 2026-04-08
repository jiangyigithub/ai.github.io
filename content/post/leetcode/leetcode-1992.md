---
title: 1992. Find All Groups of Farmland
date: 2024-04-20 15:31:49+0800
description: Count the number of farmlands
tags: 
    - Matrix
    - DFS
categories:
    - LeetCode
math: true
---

Given a matrix where its grid component representing islands and forests, count the number of farmlands.

# Intuition
Start from the top left coordinate of the farmland, Use DFS to find the bottom right coordinate of the farmland.

# Approach
We define the possible directions as `go_right` and `go_down` respectively, we iterate all grids, if it is an grid of the farmland,
then we use DFS to find the bottom right coordinate of the current farmland, and then we mark the farmland as visited and store the coordinates.

# Complexity
- Time complexity: iterate all grids once
$$O(mn)$$

- Space complexity: number of farmlands
$$O(mn)$$

# Code
```c++
class Solution {
    vector<vector<int>> dirs{{0, 1}, {1, 0}};
public:
    void dfs(vector<vector<int>>& land, int i, int j, vector<int> &bottom_right) {
        land[i][j] = 0;
        bool reach_end = true;
        for (const auto &dir: dirs) {
            int row = i + dir[0], col = j + dir[1];
            if (0 <= row && row < land.size() &&
                0 <= col && col < land[0].size() &&
                land[row][col] == 1) {
                reach_end = false;
                dfs(land, row, col, bottom_right);
            }
        }
        if (reach_end && bottom_right.empty()) {
            bottom_right = vector<int>{i, j};
        }
    }

    vector<vector<int>> findFarmland(vector<vector<int>>& land) {
        vector<vector<int>> result;
        for (int i = 0; i < land.size(); ++i) {
            for (int j = 0; j < land[0].size(); ++j) {
                if (land[i][j] == 0)    continue;
                vector<int> bottom_right;
                dfs(land, i, j, bottom_right);
                result.push_back(vector<int>{i, j, bottom_right[0], bottom_right[1]});
            }
        }
        return result;
    }
};
```
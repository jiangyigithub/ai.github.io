---
title: 623. Add One Row to Tree
date: 2024-04-14 21:06:34+0800
description: Insert a layer to a binary tree.
tags: 
    - Tree
    - BFS
categories:
    - LeetCode
math: true
---

# Intuition
Just list all nodes with height `height - 1` and insert a new layer with the given rules.

# Approach
Use a `queue` to store all nodes with the same height, and use BFS to update the nodes,
once we reach the height `height - 1`, we add a new layer with the given `val` for each `node`:
1. Create a new left child with the given `(val, node->left, nullptr)`
2. Create a new right child with the given `(val, nullptr, node->right)`

# Complexity
- Time complexity: iterate all nodes of height `height - 1`
$$O(n)$$

- Space complexity: storing all nodes of height `height - 1`
$$O(n)$$

# Code
```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    TreeNode* addOneRow(TreeNode* root, int val, int depth) {
        if (depth == 1) {
            TreeNode* new_root = new TreeNode(val, root, nullptr);
            return new_root;
        }
        queue<TreeNode*> q;
        q.push(root);
        --depth;
        while (--depth) {
            int size = q.size();
            for (int i = 0; i < size; ++i) {
                auto node = q.front();
                if (node->left)   q.push(node->left);
                if (node->right)  q.push(node->right);
                q.pop(); 
            }
        }
        while (!q.empty()) {
            auto node = q.front();
            node->left = new TreeNode(val, node->left, nullptr);
            node->right = new TreeNode(val, nullptr, node->right);
            q.pop(); 
        }
        return root;
    }
};
```
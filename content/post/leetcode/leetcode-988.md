---
title: 988. Smallest String Starting From Leaf
date: 2024-04-17 21:15:03+0800
description: find the lexicographically smallest string starting from the leaf to root node
tags: 
    - Tree
    - DFS
categories:
    - LeetCode
math: true
---

Given a binary tree with value on each node representing a lowercase letter, find the lexicographically smallest string starting from the leaf to root node

# Intuition
We use DFS to find all strings from the root node to the leaf nodes, then reverse the string and compare it withe the largest string.

# Approach
We use `current` to represent the string starting from the root node to the current node and we use `result` to store the currently best result. When we reach the leaf node, we compare the `current` with `result` and update `result`.

# Complexity
- Time complexity: iterate all nodes once.
$$O(n)$$

- Space complexity: the string is related to the height of the tree.
$$O(\log n)$$

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
    void dfs(TreeNode* root, string &current, string &result) {
        current.push_back('a' + root->val);
        // leaf node
        if (!root->left && !root->right) {
            // update result
            reverse(current.begin(), current.end());
            if (current < result)   result = current;
            reverse(current.begin(), current.end());
        } else {
            if (root->left) dfs(root->left, current, result);
            if (root->right) dfs(root->right, current, result);
        }
        current.pop_back();
    }

    string smallestFromLeaf(TreeNode* root) {
        string result(8501, 'z');
        string current;
        dfs(root, current, result);
        return result;
    }
};
```
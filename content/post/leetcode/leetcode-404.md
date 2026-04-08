---
title: 404. Sum of Left Leaves
date: 2024-04-14 21:06:34+0800
description: Sum of Left Leaves of a binary tree.
tags: 
    - Tree
    - DFS
categories:
    - LeetCode
math: true
---

Given the `root` of a binary tree, return the sum of all left leaves.


## Intuition
Use DFS to iterate over all nodes, if it is a left leaf, sum it to the result.

## Approach
For every node, we care about one thing: whether its left child is a leaf node or not. If it is, then we add it.

## Complexity
- Time complexity: iterate all nodes once.
$$ O(n) $$

- Space complexity: Store the sum.
$$ O(1) $$

## Code
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
    bool is_leaf_node(TreeNode* node) {
        if (!node)  return false;
        return !(node->left || node->right);
    }

    int sumOfLeftLeaves(TreeNode* root) {
        if (!root)  return 0;
        int result = 0;
        if (is_leaf_node(root->left)) {
            result += root->left->val;
        }
        result += sumOfLeftLeaves(root->right);
        result += sumOfLeftLeaves(root->left);
        return result;
    }
};
```
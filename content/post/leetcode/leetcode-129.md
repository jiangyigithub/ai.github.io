---
title: 129. Sum Root to Leaf Numbers
date: 2024-04-14 21:06:34+0800
description: Sum of paths from root to leaf
tags: 
    - Tree
    - DFS
categories:
    - LeetCode
math: true
---

concatenate the digit of path from the root to the leaf, and sum over the concatenated numbers.


# Intuition
Use DFS to find all the paths, use a `num` variable to store the number concatenated, then sum over `num`.

# Approach
We use `num` to represent the number from the root to the current node, if the current node is a leaf node,
we add `num` to `sum`. Finally, we return `sum` as the result.

# Complexity
- Time complexity: iterate all nodes once.
$$O(n)$$ 

- Space complexity:
$$O(1)$$

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
    void dfs(TreeNode *root, int &sum, int &num){
        if (!root) {
            return;
        }
        num = num * 10 + root->val;
        if (!root->left && !root->right){
            sum += num;
        }else{
            dfs(root->left, sum, num);
            dfs(root->right, sum, num);
        }
        num /= 10;
    }

    int sumNumbers(TreeNode* root) {
        int sum = 0;
        int num = 0;
        dfs(root, sum, num);
        return sum;
    }
};
```
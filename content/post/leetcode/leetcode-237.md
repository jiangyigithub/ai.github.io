---
title: 237. Delete Node in a Linked List
date: 2024-05-05 20:13:45+0800
description: Delete a node  from a linked list with accessing the head of the linked list
tags: 
    - Linked List
    - Medium
categories:
    - LeetCode
math: true
---

Given a linked list and the node to be deleted, delete the node without accessing the head of the linked list

# Intuition
This deletion is same as deleting a node from an array.

# Approach
We use two pointers, `pre` and `node` to represent the previous and current node of the linked list, and we update the value of `pre` and `node` at each iteration. Finally, we delete the last node in the linked list.

# Complexity
- Time complexity: iterate the linked list once.
$$O(n)$$

- Space complexity: no extra space needed.
$$O(1)$$

# Code
```c++
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    void deleteNode(ListNode* node) {
        ListNode* pre = nullptr;
        while (node->next) {
            node->val = node->next->val;
            pre = node;
            node = node->next;
        }
        pre->next = nullptr;
    }
};
```


# References
- [Leetcode](https://leetcode.com/problems/delete-node-in-a-linked-list/description/)
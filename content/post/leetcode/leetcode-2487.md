---
title: 2487. Remove Nodes From Linked List
date: 2024-05-06 19:47:28+0800
description: Remove all nodes from linked list if there exists a node in the right hand side of the node has a greater value.
tags: 
    - Linked List
    - Monotonic Stack
    - Medium
categories:
    - LeetCode
math: true
---

Given a linked list, we are required to remove some nodes, such that for each node in the result linked list, the value of the node is the greatest from the node to the end of the linked list.


# Intuition
We construct the result linked list from right to left, that is, the last node in the linked list is kept, then the pointer goes from right to left util there is a node with greater value. This process can be done via post traversal as tree.


## Complexity
- Time complexity: iterate the list once.
$$O(n)$$ 

- Space complexity: use a set to store numbers.
$$O(n)$$


## Code
```c++
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution {
public:
    void reverse_traversal(ListNode* head, stack<int> &s) {
        if (!head)  return;
        reverse_traversal(head->next, s);
        if (s.empty() || head->val >= s.top()) {
            s.push(head->val);
        }
    }

    ListNode* removeNodes(ListNode* head) {
        stack<int> s;
        reverse_traversal(head, s);
        ListNode* dummyhead = new ListNode(0, nullptr);
        ListNode* cur = dummyhead;
        while (!s.empty()) {
            cur->next = new ListNode(s.top(), nullptr);
            cur = cur->next;
            s.pop();
        }
        return dummyhead->next;
    }
};
``

# References
- [Leetcode](https://leetcode.com/problems/remove-nodes-from-linked-list/description/)
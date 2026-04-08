---
title: 2816. Double a Number Represented as a Linked List
date: 2024-05-06 19:47:28+0800
description: Double a Number Represented as a Linked List
tags: 
    - Linked List
    - Stack
    - Medium
categories:
    - LeetCode
math: true
---

Given a linked list representing a non-negative integer, we are required to double this integer and convert it back to a linked list.

# Intuition
We can just simulate the process, that is:

1. Retrieve the integer represented by the linked list
2. Double the integer
3. Construct the result linked list from the result integer.

# Approach
We can just simulate the process as above. However, if the integer is very large, retrieve the integer may cause overflow. To simplify this process, we can use a stack to store the digits of the original integer, then we double the integer by operating on the top of the stack.

In each step, we pop an element from the stack, doubling it and adding it with the carry digit. Then we construct the result linked list with inserting the new node in the front of the head.


# Complexity

- Time complexity: iterate all digits twice.
$$O(n)$$

- Space complexity: store the result linked list.
$$O(n)$$

# Code

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
    ListNode* doubleIt(ListNode* head) {
        stack<int> s;
        ListNode* cur = head;
        while (cur) {
            s.push(cur->val);
            cur = cur->next;
        }
        int carry = 0;
        while (!s.empty()) {
            int num = s.top() * 2 + carry;
            ListNode* pre =  new ListNode(num % 10, cur);
            carry = num / 10;
            cur = pre;
            s.pop();
        }
        if (carry) {
            ListNode* pre =  new ListNode(carry, cur);
            cur = pre;
        }
        return cur;
    }
};
```

# References
- [Leetcode](https://leetcode.com/problems/double-a-number-represented-as-a-linked-list/description)

---
title: 752. Open the Lock
date: 2024-04-22 19:01:23+0800
description: Sum of paths from root to leaf
tags: 
    - String
    - BFS
categories:
    - LeetCode
math: true
---

Given a four-digit string, change one digit (plus or minus 1) at a time, find the minimum number of steps to go from the source to target without passing through invalid states.

# Intuition
We can image this as a graph path finding problem, where we need to find a path from the `source` to the `target` with the minimum number of steps.

# Approach
We use BFS to solve this problem. The graph is constructed as follows: each possible state is a node of the graph, such as `"1234"`, `"4567"`, each operation defines an edge between two nodes, for example, we can rotate third digit of `"1234"` to obtain `"1244"`, since there are two possible directions and four digits, each node has $2^4=16$ adjacent nodes. We keep track of visited nodes and add them to `deadends` since there are no difference between them.

# Complexity
- Time complexity: there are at most $10^4=10000$ nodes and we visited all nodes at most once.
$$O(1)$$

- Space complexity: there are at most $10^4=10000$ nodes.
$$O(1)$$

# Code
```c++
class Solution {
public:
    vector<string> get_ajacent_nodes(const string &s) {
        vector<string> result;
        for (int i = 0; i < s.size(); ++i) {
            string s1 = s;
            string s2 = s;
            if ('1' <= s[i] && s[i] <= '9') {
                s1.replace(i, 1, string(1, s[i] + 1));
                s2.replace(i, 1, string(1, s[i] - 1));
            } else if (s[i] == '9') {
                s1.replace(i, 1, "0");
                s2.replace(i, 1, "8");
            } else {
                s1.replace(i, 1, "1");
                s2.replace(i, 1, "9");
            }
            result.push_back(s1);
            result.push_back(s2);
        }
        return result;
    }

    int openLock(vector<string>& deadends, string target) {
        unordered_set<string> set_deadends(deadends.begin(), deadends.end());
        queue<string> q;
        q.push("0000");
        int result = 0;
        while (!q.empty()) {
            int size = q.size();
            for (int i = 0; i < size; ++i) {
                string node = q.front();
                q.pop();
                // dead end check
                if (set_deadends.find(node) != set_deadends.end())  continue;
                // target check
                if (node == target) return result;
                // mark as visited
                set_deadends.insert(node);
                const vector<string> &adjacent_nodes = get_ajacent_nodes(node);
                for (const string &adjacent_node: adjacent_nodes) {
                    q.push(adjacent_node);
                }
            }
            ++result;
        }
        return -1;
    }
};
```
---
title: 592. Fraction Addition and Subtraction
date: 2024-08-23 20:16:54+0800
description: Compute the value from a valid fraction string
tags: 
    - String
    - Medium
categories:
    - LeetCode
math: true
---

Given a string expression representing an expression of fraction addition and subtraction, return the calculation result in string format.

# Intuition

Simulate the calculation of fraction addition and subtraction process.

# Approach

First, to simplify addition and subtraction, we move the `-` to numerator part, for example, `3/4-2/3` becomes `3/4+(-2)/3`, this step makes all operations become addition.

Second, we use a initial value `0/1` to record result to prevent from parsing the first fraction. Thus `3/4-2/3` is actually equivalent to `0/1+3/4-2/3`.

Now, in each loop, we record the sign, the numerator, the denominator. Then we compute the result with the previous result with the formula

$$
\frac{a}{b} + \frac{c}{d} = \frac{ad-bc}{bd}
$$

after computation, we use `gcd()` function to make the resulted fraction irreducible.

# Complexity

- Time complexity: Iterate the string once.
$$O(n)$$

- Space complexity: No extra spaces needed.
$$O(1)$$

# Code

```c++
class Solution {
public:
    string fractionAddition(string expression) {
        int n = expression.size();
        int numerator1 = 0, dominator1 = 1;
        int numerator2 = 0, dominator2 = 1;
        int sign = 1;
        for (int i = 0; i < n;) {
            if (expression[i] == '-') {
                sign = -1;
                ++i;
                continue;
            }
            if (expression[i] == '+') {
                sign = 1;
                ++i;
                continue;
            }
            numerator2 = 0;
            while (i < n && '0' <= expression[i] && expression[i] <= '9') {
                numerator2 = numerator2 * 10 + (expression[i] - '0');
                ++i;
            }
            numerator2 *= sign; // move sign to numerator
            sign = 1;   // reset sign
            ++i;    // division operator
            dominator2 = 0;
            while (i < n && '0' <= expression[i] && expression[i] <= '9') {
                dominator2 = dominator2 * 10 + (expression[i] - '0');
                ++i;
            }
            numerator1 = (numerator1 * dominator2 + numerator2 * dominator1);
            dominator1 = dominator1 * dominator2;
            if (numerator1 && dominator1) {
                int gcd_num = gcd(numerator1, dominator1);
                numerator1 = numerator1 / gcd_num;
                dominator1 = dominator1 / gcd_num;
            }
        }
        // corner case
        if (numerator1 == 0) {
            return "0/1";
        }

        return to_string(numerator1) + "/" + to_string(dominator1);
    }
};
```

# Reference

- [leetcode 592](https://leetcode.com/problems/fraction-addition-and-subtraction/description)

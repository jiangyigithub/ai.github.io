---
title: Understanding Sigmoid Loss in SigLip
date: 2025-03-28 14:55:50+0800
description: Understanding Sigmoid Loss in SigLip
tags: 
    - loss
categories: 
    - MLLM
    - Machine Learning 
math: true
---

## Introduction

A simple note to understand Sigmoid Loss in SigLip [^1]. Supported by DeepSeek[^2]

## Binary cross entropy loss

Suppose we want to solve the binary classification problem, with label $y\in\{0, 1\}$, a common option is to use binary cross entropy loss:

$$\mathcal{L}(x, y) = -[y\log (\sigma(z)) + (1-y)\log (1-\sigma(z))]$$

where $z=f_\theta(x)$ is the logits predicted by our model $f_\theta$, and $\sigma$ is the sigmoid function:

$$\sigma(z) := \frac{1}{1 + e^{-z}}$$

Let $\sigma(\cdot)$ be the sigmoid function, then we have:

$$
\sigma(-z) = \frac{1}{1 + e^{z}} = \frac{e^{-z}}{1 + e^{-z}} = 1 - \frac{1}{1 + e^{-z}} = 1- \sigma(z)
$$

Now we substitute $\sigma(-z)=1-\sigma(z)$ into the loss function, we obtain:

$$\mathcal{L}(x, y) = -[y\log (\sigma(z)) + (1-y)\log (\sigma(-z))]$$

Note that $y\in\{0, 1\}$ thus for each instance, there are two cases:

- If $y=0$, then $\mathcal{L}(x, y) =-\log (\sigma(-z))$
- If $y=1$, then $\mathcal{L}(x, y) =-\log (\sigma(z))$

Now we want to use a unified expression to express these two cases. Note that this requires fitting a curve that passes two points $(0, -1)$ and $(1, 1)$. The simplest curve is a straight line $y=2x-1$. So, we can further simplify the loss expression into:

$$\mathcal{L}(x, y) = -\log\left[\sigma((2y-1)z)\right]$$

## Sigmoid Loss in SigLip

Now we recall the sigmoid loss in SigLip:

$$\mathcal{L}(\{\bm{x}, \bm{y}\}_{i=1}^N)=-\frac{1}{N}\sum_{i=1}^N\sum_{j=1}^N\log \frac{1}{1+\exp\left[z_{ij}(-t\bm{x}_i\cdot \bm{y_j}+b)\right]}$$

where $t, b$ are learnable parameters, and $z_{ij}=1$ if $i=j$ and $z_{ij}=-1$ otherwise.

To understand Sigmoid loss, notice that $z_{ij}=2\mathbb{I}_{i=j}-1$, which exactly matches the form we derived earlier.

## Why Use Sigmoid Loss?

1. More stable: avoids $\log 0$.
2. More efficient: Compute Sigmoid once.
3. More Precise: one line of code without condition checking.

# References

[^1]: [SigLip](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhai_Sigmoid_Loss_for_Language_Image_Pre-Training_ICCV_2023_paper.pdf)
[^2]: [DeepSeek](https://chat.deepseek.com/)

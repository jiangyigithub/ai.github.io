---
title: Fix Point Theorem
description: 不动点定理
date: 2026-03-09 17:16:02+08:00
math: true
categories:
    - Math
---


Fix Point Theorem, 即不动点定理，是泛函分析中的基本工具，被广泛应用于非线性函数的分析。

在介绍不动点定理之前，我们先介绍两个概念

首先是不动点的概念。

**Definition**

对于函数 $f:\mathbb{R}^n\to\mathbb{R}^n$, 如果一个点 $x^*\in\mathbb{R}^n$ 满足

$$
f(x^*)=x^*
$$

则我们称 $x^*$ 是函数 $f$ 的不动点。

接下来是 contraction mapping 的概念

**Definition**

对于函数 $f:\mathbb{R}^n\to\mathbb{R}^n$, 如果存在 $\gamma\in(0,1)$ 满足

$$
\|f(x_1)-f(x_2)\| \leq \gamma \|x_1-x_2\|,\forall\ x_1,x_2\in\mathbb{R}^n
$$

则我们称 $f$ 是一个 contraction mapping. 这里 $\|\cdot\|$ 是一个 matrix norm.

接下来，我们介绍不动点定理

**Theorem**

给定 equation $x=f(x)$, 其中 $f:\mathbb{R}^n\to\mathbb{R}^n$, 如果 $f$ 是一个 contraction mapping, 则 $f$ 具有如下性质

1. Existence: 存在 fixed point $x^*\in\mathbb{R}^n$ 满足 $f(x^*)=x^*$.
2. Uniqueness: fixed point $x^*$ 唯一。
3. Algorithm: 对任意 $x^0\in\mathbb{R}^n$, 使用迭代算法 $x_{k+1}=f(x_k)$ 产生的序列 $\{x_k\}_{k=0}^{\infty}$ 收敛到 fixed point $x*$, 且收敛速度为指数级。

证明需要用到柯西列的概念。

**Definition**

一个序列 $x_1,x_2,\dots$ 被称为柯西列 (Cauchy sequence) 当且仅当对任意 $\epsilon>0$, 都存在 $N>0$, 使得

$$
\|x_m-x_n\| <\epsilon,\forall m, n>N
$$

柯西列的一个重要性质为柯西列一定是收敛列。

**证明**

我们首先证明由 $x_k=f(x_{k=1})$ 产生的序列 $\{x_k\}_{k=1}^{\infty}$ 是收敛的，我们通过证明序列 $\{x_k\}_{k=1}^{\infty}$ 是一个柯西列来证明这一点。

注意到 $f$ 是一个 contraction mapping, 因此

$$
\|x_{k+1}-x_k\| = \|f(x_k)-f(x_{k-1})\|\leq \gamma \|x_k-x_{k-1}\|
$$

迭代下去，我们就得到

$$
\|x_{k+1}-x_k\|\leq \gamma \|x_k-x_{k-1}\|\leq\cdots\leq \gamma^k \|x_1-x_{0}\|
$$

现在我们证明序列 $\{x_k\}_{k=1}^{\infty}$ 是一个柯西列：

$$
\begin{aligned}
\|x_m-x_n\| &= \|x_m-x_{m-1}+x_{m-1}-\cdots-x_{n+1}+x_{n+1}-x_n\|\\
&\leq \sum_{i=n}^{m-1}\|x_{i+1}-x_i\|\\
&\leq \sum_{i=n}^{m-1}\gamma^i \|x_{1}-x_0\|\\
&\leq \frac{\gamma^n}{1-\gamma}\|x_{1}-x_0\|.
\end{aligned}
$$

从而，序列 $\{x_k\}_{k=1}^{\infty}$ 是一个柯西列, 因此也是一个收敛列。

接下来，我们证明 $x^*=\lim_{k\to\infty}x_k$ 是 $f(x)$ 的不动点，注意到

$$
\|f(x_k)-x_k\| = \|x_{k+1}-x_k\|\leq \gamma^k\|x_1-x_0\| \to 0, k\to\infty
$$

我们有 $\lim_{k\to\infty}f(x_k)=\lim_{k\to\infty}x_k$ , 由于 contraction mapping 一定是连续的，因此我们就可以得到 $f(x^*)=x^*$.

然后，我们证明不动点唯一。假设还存在一个另外一个不动点 $x'\neq x^*$ 满足 $f(x')=x'$, 那么

$$
\|x'-x^*\| = \|f(x')-f(x')\| \leq \gamma \|x'-x^*\|
$$

由于 $\gamma\in(0,1)$, 因此上述等式当且仅当 $\|x'-x^*\|=0$, 这与前面假设矛盾，因而不动点是唯一的

最后，我们证明 $x_{k+1}=f(x_k)$ 这个算法的收敛速度为指数级，注意到

$$
\|x^*-x_n\| = \lim_{m\to\infty}]\|x_m-x_n\| \leq \frac{\gamma^n}{1-\gamma}\|x_1-x_0\|
$$

因为 $\gamma <1$, 因此收敛速度为指数级

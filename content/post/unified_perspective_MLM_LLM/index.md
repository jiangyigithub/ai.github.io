---
title: Unified perspective on dLLM and LLM
description: MLE和KL divergence之间的等价性推导
date: 2025-06-28 15:02:09+0800
tags: 
    - diffusion
categories:
    - LLM
math: true 
---


# Introduction

在上一篇blog里，我们介绍了MLE和KL minimization的等价性。在这篇blog里，我们将要基于这个等价性，推导masked diffusion LLM, autoregressive LLM, any-order diffusion LLM之间的等价性。最终，我们发现，这几种建模方式本质上都是一致的。

# Preliminary

对于 $\bm{x}=(x_1,\dots,x_L)\in\mathbb{R}^L$, 基于概率的链式法则，我们有

$$
p(\bm{x}) = \prod_{i=1}^Lp(x_i\mid x_{<i})
$$

1D的类别分布，对于随机变量 $X\in\{1,\dots, K\}$以及概率分布 $\bm{p}=(p_1,\dots,p_K)$ 我们定义类别分布为 $\mathrm{Cat}(\bm{x};\bm{p})$, 其中 $\bm{x}\in\mathbb{R}^K$是一个向量，其PMF定义为：

$$
f(\bm{x}; \bm{p}) = \prod_{i=1}^Kp_i^{[x=i]}
$$

这里 $[x=i]$ 当 $x=i$时为$1$，否则为 $0$.

一个比较好理解的例子是骰子，骰子有$K=6$种可能性，每一面出现的可能性都是 $p_i=1/6$. 当 $\bm{x}$是one-hot向量是，其代表了出现某一面的概率。

Discrete-Time Discrete Markov Chain. 接下来我们考虑离散时间离散Markov链，我们现在的随机变量不仅与状态 $\{1,\dots,K\}$ 还与时间有关，我们记 $X_n$ 为 $n$ 时刻的状态分布。这样，我们可以规定不同时刻之间的状态转换矩阵：

$$
Q_{ij} = \mathrm{Pr}(x_{n+1}=j\mid x_n=i)
$$

## Categorical distribution

# autoregressive LLM

首先是我们最常见的自回归大语言模型，给定文本语料 $\bm{x}=(x_1,\dots,x_L)\sim p_{data}$， 大语言模型的目标在于求解最大似然估计

$$
\begin{aligned}
   \theta^* &=  \arg\max_{\theta}\mathbb{E}_{\bm{x}\sim p_{data}}[\log p(\bm{x}\mid \theta))]\\
   &=  \arg\max_{\theta}\mathbb{E}_{\bm{x}\sim p_{data}}\left[\log \left(\prod_{i=1}^Lp(x_i\mid x_{<i},\theta )\right) \right]\\
    &=   \arg\max_{\theta}\mathbb{E}_{\bm{x}\sim p_{data}}\left[\sum_{i=1}^L\log p(x_i\mid x_{<i},\theta ) \right]
\end{aligned}
$$

这里我们使用了链式法则.

# Any order autoregressive LLM

接下来，我们证明Any order autoregressive LLM的目标函数等价于求解MLE。
令 $S_D$ 为在集合 $\{1,\dots, D\}$ 上所有可能的permutation，令 $\sigma\in S_D$ 为一个其中的一个permutation，则我们有

$$
p(\sigma) = \frac{1}{D!}
$$

因此

$$
p(x) = \sum_{\sigma\in S_d}p(x,\sigma) = \sum_{\sigma\in S_D}p(x\mid \sigma)p(\sigma) = \frac{1}{D!}\sum_{\sigma\in S_D}p(x\mid \sigma)=\mathbb{E}_{\sigma\sim U(S_D)}[p(x\mid \sigma)]
$$

这里 $U(\cdot)$ 代表均匀分布，从而我们有

$$
\log p(x) = \log \mathbb{E}_{\sigma\sim U(S_D)}[p(x\mid \sigma)] \geq \mathbb{E}_{\sigma\sim U(S_D)}[\log p(x\mid \sigma)]
$$
这里我们使用了Jensen不等式

那么对于any-order autoregressive model, 我们有：

$$
\log p(x) \geq \mathbb{E}_{\sigma\sim U(S_D)}\sum_{t=1}^D \log p(x_{\sigma(t)}\mid x_{\sigma(<t)})
$$

也就是说，对于any-order autoregressive model，我们的目标函数是MLE的一个下界。

# Masked Language modeling

# Discrete Diffusion Model

# Any-order diffusion LLM

# Masked diffusion LLM

# Reference

- [Structured Denoising Diffusion Models in Discrete State-Spaces](https://arxiv.org/pdf/2107.03006)

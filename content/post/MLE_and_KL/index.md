---
title: Relationship between MLE and KL divergence
description: MLE和KL divergence之间的等价性推导
date: 2025-06-27 11:35:33+0800
tags: 
    - MLE
    - KL divergence
categories:
    - Machine Learning
math: true 
---

# MLE

最大似然估计，即MLE (maximum likelihood estimation), 是一个估计参数分布的方法，其核心思想是：模型的参数，应该让观察样本出现的概率最大。

假设我们有一个参数分布 $p(x\mid \theta)$, 其中 $\theta$ 是参数，如正态分布中的均值和方差。我们从$p(x\mid \theta)$进行采样得到 $i.i.d.$ 的数据 $X=\{x_1,\dots,x_n\}$.

似然函数 (likelihood function) 定义为给定数据 $X$ 的联合分布，即：

$$
\mathcal{L}(\theta\mid X) = P(X\mid \theta)
$$

由于 $X=\{x_1,\dots,x_n\}$ 是 $i.i.d.$, 因此，我们可以将上式改写为：

$$
\mathcal{L}(\theta\mid X) = \prod_{i=1}^n p(x_i\mid \theta)
$$

这样我们的优化目标就是

$$
\begin{aligned}
 \theta_{MLE}^* &= \arg\max_{\theta} \mathcal{L}(\theta\mid X)\\
 &= \arg\max_{\theta} \prod_{i=1}^n p(x_i\mid \theta)\\
 &= \arg\max_{\theta} \log\prod_{i=1}^n p(x_i\mid \theta)\\
 &=\arg\max_{\theta} \sum_{i=1}^n \log p(x_i\mid \theta)\\
\end{aligned}
$$

即

$$
\theta_{MLE}^* = \arg\max_{\theta} \sum_{i=1}^n \log p(x_i\mid \theta)
$$

# KL divergence

KL divergence 用于衡量概率分布 $Q(x)$ 到概率分布 $P(x)$ 的不同程度，我们可以将其理解为：如果我们用 $Q(x)$来替换 $P(x)$, 会有多大的信息损失？

连续概率分布的KL divergence的定义如下

$$
D_{KL}(P\mid\mid Q) =\int P(x)\log\left(\frac{P(x)}{Q(x)}\right)dx
$$
离散概率分布的KL divergence定义如下

$$
D_{KL}(P\mid\mid Q) = \sum_{x} P(x)\log\left(\frac{P(x)}{Q(x)}\right)
$$

KL divergence有两个关键性质：

1. 非负性：$D_{KL}(P\mid\mid Q)\geq0$, 且 $D_{KL}(P\mid\mid Q)=0$ 当且仅当 $P(x)=Q(x)$ 对任意 $x$成立
2. 非对称性： 一般情况下，$D_{KL}(P\mid\mid Q)\neq D_{KL}(Q\mid\mid P)$.

# MLE和KL Divergence的等价性

我们假设 $p_{data}(x)$ 是数据$X$的真实分布， 我们现在需要找到合适的参数 $\theta$ 以及其对应的分布 $p(x\mid \theta)$ 来近似 $p_{data}(x)$, 此时我们可以用KL Divergence作为我们的目标函数，即

$$
\theta_{KL} = \arg\min_{\theta}D_{KL}(p_{data}(x)\mid\mid p(x\mid \theta))
$$

我们将上面的式子进行展开得到

$$
\begin{aligned}
\theta_{KL}^* &= \arg\min_{\theta}D_{KL}(p_{data}(x)\mid\mid p(x\mid \theta))\\
&= \arg\min_{\theta} \int p_{data}(x)\frac{p_{data}(x)}{p(x\mid \theta)} dx\\
&= \arg\min_{\theta}\int p_{data}(x)\log p_{data}(x) dx - \int p_{data}(x)\log p(x\mid \theta)dx \\
&= \arg\min_{\theta} - \int p_{data}(x)\log p(x\mid \theta)dx \\
&= \arg\max_{\theta} \int p_{data}(x)\log p(x\mid \theta)dx
\end{aligned}
$$

实际上，真实的数据分布 $p_{data}(x)$ 是未知的，我们只有从 $p_{data}(x)$ 采样得到的一批数据 $X=\{x_1,\dots,x_n\}\sim p_{data}(x)$.

基于大数定律，我们有

$$
\frac{1}{n}\sum_{i=1}^n\log p(\theta_i\mid \theta)=\mathbb{E}_{x\sim p_{data}}[\log p(x\mid \theta)] = \int p_{data}(x)\log p(x\mid \theta)dx, n\to \infty
$$

这样，最大似然估计就与最小化KL divergence构建起了联系：

$$
\begin{aligned}
\theta_{MLE}^*&=\arg\max_{\theta} \sum_{i=1}^n \log p(x_i\mid \theta)\\
&= \arg\max_{\theta} \int p_{data}(x)\log p(x\mid \theta)dx\\
&= \theta_{KL}^*, n\to\infty.
\end{aligned}
$$

也就是说，当采样样本足够多的时候，最大似然估计和最小KL divergence是等价的。

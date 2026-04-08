---
title: (RL series 2) Bellman Equation
description: 本节介绍bellman equation相关概念
date: 2026-03-18 17:39:45+08:00
math: true
tags:
    - tutorial
categories:
    - Math
    - RL
---



## Bellman Equations

本节中，我们将要定义 value function, Q-function 以及这两个函数与 policy 之间的关系，这是我们介绍不同 RL 算法的基础。这一节需要使用到 [不动点定理](https://maosong.website/p/fix-point-theorem/).

我们定义 Value function 如下：

$$
V^{\pi}(s) = \mathbb{E}^{\pi}[G_0\mid s_0=s]
$$

value function 的具体含义为：*agent 从当前状态 $s$ 出发，一直遵循当前策略 $\pi$， 最后获取到的 expected discounted  return*.

state-action value function, 或者 Q-value function 定义如下：

$$
Q^{\pi}(s,a) = \mathbb{E}^{\pi}[G_0\mid s_0=s, a_0=a]
$$

其具体含义为：*agent 从当前状态 $s$ 出发，执行 action $a$,  再遵循当前策略 $\pi$, 最后获取到的 expected discounted return*.

> 注意
> 这里为了方便，我们令 $V^{\pi}(\langle term\rangle)=0$, $Q^{\pi}(\langle term\rangle,a)=0,\forall a\in\mathcal{A}$.

由全概率公式，我们有

$$
\mathbb{E}[G_0\mid s_0=s] = \sum_{a\in\mathcal{A}}\mathbb{E}\left[G_0\mid s_t=s, a_0=a\right]\pi(a\mid s_0=s)
$$

即

$$
\boxed{V^{\pi}(s)  = \mathbb{E}_{a_0\sim \pi(\cdot\mid s_0)}[Q^{\pi}(s,a)]}
$$

一般来说，我们会使用 value function 和 Q-value function 的如下递推性质：

$$
\boxed{
\begin{aligned}V^\pi(s) &= \mathbb{E}_{a_0\sim \pi(\cdot\mid s),\ (r_0,s_0)\sim p(\cdot,\cdot\mid s,a_0)}[r_0 + \gamma V^\pi(s_1)\mid s_0=s]\\
Q^\pi(s,a) &= \mathbb{E}_{(r,s')\sim p(\cdot,\cdot\mid s,a),\ a'\sim \pi(\cdot\mid s')}[r + \gamma Q^\pi(s',a')\mid s,a]
\end{aligned}}
$$

证明需要利用 Markov property:

$$
\begin{aligned}
V^{\pi}(s) &= \mathbb{E}^{\pi}\left[r_0 + \gamma\sum_{t=1}^{T-1}\gamma^{t-1}r_t\mid s_0=s\right]\\
&= \mathbb{E}^{\pi}[r_0+\gamma G_1\mid s_0=s]\\
&= \mathbb{E}_{a_0\sim \pi(\cdot\mid s,\ (r_0,s_0)\sim p(\cdot,\cdot\mid s,a_0))}\left[\mathbb{E}^\pi[r_0 + \gamma G_1\mid s_0,a_0,r_0,s_1]\mid s_0=s\right]\\
&= \mathbb{E}_{a_0\sim \pi(\cdot\mid s,\ (r_0,s_0)\sim p(\cdot,\cdot\mid s,a_0))}\left[r_0 + \gamma \mathbb{E}^\pi[G_1\mid s_1]\mid s_0=s\right]\\
&= \mathbb{E}_{a_0\sim \pi(\cdot\mid s,\ (r_0,s_0)\sim p(\cdot,\cdot\mid s,a_0))}\left[r_0 + \gamma V^\pi[s_1)\mid s_0=s\right]
\end{aligned}
$$

对于 $Q^\pi(s,a)$ 的推导同理。

我们接下来介绍 RL 的核心：Bellman equation Theorem,  其阐述了策略 $\pi$ 与 value function 所满足的充分必要条件。

**Theorem**

> 令 $\pi$ 为一个策略，假设 $\gamma\in(0,1)$, $|\mathcal{S}|<\infty$ 以及 $|r|\leq R<\infty, \mathrm{a.s.}$. 那么 $\pi$ 对应的 value function $V^\pi:\mathcal{S}^+\to\mathbb{R}$ 存在，且满足 Bellman equation:

$$
\boxed{
V^\pi(s) = \mathbb{E}_{a\sim \pi(\cdot\mid s),\ (r,s')\sim p(\cdot,\cdot\mid s,a_0))}\left[r+\gamma V(s')\mid s\right]
}
$$

> 反之，如果存在函数 $V:\mathcal{S}^+\to\mathbb{R}$ 满足 Bellman equation, 则 $V=V^\pi$.

**证明**

由 $V^\pi$ 定义，我们有

$$
\left|\sum_{t=0}^{\infty}\gamma^tr_t\right|\leq \sum_{t=0}^\infty \gamma^t R=\frac{R}{1-\gamma}<\infty
$$

因此，$\sum_{t=0}^{\infty}\gamma^tr_t$ 是一个绝对收敛的序列，从而其期望存在且有界。

对于 $V^\pi(s)=\mathbb{E}^{\pi}[G_0\mid s_0=s]$ , 我们有

$$
\begin{aligned}
V^\pi(s)&=\mathbb{E}^{\pi}\left[G_0\mid s_0=s\right]\\
&= \mathbb{E}^{\pi}\left[r_0+\gamma G_1\mid s_0=s\right]\\
&=\mathbb{E}_{a\sim\pi(\cdot\mid s), (r,s')\sim p(\cdot,\cdot\mid s,a_0)}\left[\mathbb{E}^\pi\left[r_0+\gamma G_1\mid s_0,a_0,r_0,s'\right]\right]\\
&= \mathbb{E}_{a\sim\pi(\cdot\mid s), (r,s')\sim p(\cdot,\cdot\mid s,a_0)}\left[r_0+\gamma\mathbb{E}^\pi\left[ G_1\mid s_0,a_0,r_0,s'\right]\right]\\
&= \mathbb{E}_{a\sim\pi(\cdot\mid s), (r,s')\sim p(\cdot,\cdot\mid s,a_0)}\left[r_0+\gamma\mathbb{E}^\pi\left[ G_1\mid s_1\right]\right]\\
&=\mathbb{E}_{a\sim\pi(\cdot\mid s'), (r,s')\sim p(\cdot,\cdot\mid s,a_0)}\left[r_0+V^{\pi}(s')\right]
\end{aligned}
$$

其中第 5 个等式为 Markov property.

最后，我们证明方程解的唯一性，假设存在函数 $V:\mathcal{S}^+\to\mathbb{R}$ 满足 Bellman equation. 我们定义 Bellman 算子 $\mathcal{T}^\pi$ 为

$$
(\mathcal{T}^\pi V)(s) := \mathbb{E}_{a\sim \pi(\cdot\mid s),\ (r,s')\sim p(\cdot,\cdot\mid s,a_0))}\left[r+\gamma V(s')\mid s\right]
$$

我们证明该算子是一个 contraction mapping, 考虑范数 $\|V\|_\infty = \max_{s\in\mathcal{S}} |V(s)|$, 我们有

$$
\begin{aligned}
\left|(\mathcal{T}^\pi V_1)(s)-(\mathcal{T}^\pi V_2)(s)\right| &= \left|\mathbb{E}_{a\sim\pi(\cdot\mid s'), (r,s')\sim p(\cdot,\cdot\mid s,a_0)}\left[\gamma V_1^{\pi}(s')-\gamma V_2^{\pi}(s')\right]\right|\\
&\leq \gamma \mathbb{E}_{a\sim\pi(\cdot\mid s'), (r,s')\sim p(\cdot,\cdot\mid s,a_0)}\left|V_1^{\pi}(s')-V_2^{\pi}(s')\right|\\
&\leq \gamma \max_{s''\in\mathcal{S}}\left|V_1(s'')-V_2(s'')\right|\\
&= \gamma \left\|V_1-V_2\right\|_\infty
\end{aligned}
$$

上式对于任意 $s\in\mathcal{S}$ 都成立，因此

$$
\left\|\mathcal{T}^\pi V_1-\mathcal{T}^\pi V_2\right\|_\infty \leq \gamma \left\|V_1-V_2\right\|_\infty
$$

由于 $\gamma < 1$, 从而 $\mathcal{T}^\pi$ 是一个 contraction mapping.

根据不动点定理，已知 $V^\pi$ 是一个不动点，而不动点唯一，因此我们有 $V=V^\pi$. $\blacksquare$

同理，我们可以推导出关于 Q-function 的 Bellman equation:

**Theorem**

> 令 $\pi$ 为一个策略，假设 $\gamma\in(0,1)$, $|\mathcal{S}|<\infty$ $|\mathcal{A}|<\infty$ 以及 $|r|\leq R<\infty, \mathrm{a.s.}$. 那么 $\pi$ 对应的 Q-function $Q^\pi:\mathcal{S}^+\times\mathcal{A}\to\mathbb{R}$ 存在，且满足 Bellman equation:

$$
\boxed{
Q^\pi(s, a) = \mathbb{E}_{(r,s')\sim p(\cdot,\cdot\mid s,a),\ a'\sim \pi(\cdot\mid s')}\left[r+\gamma Q(s', a')\mid s, a\right]
}
$$

> 反之，如果存在函数 $Q:\mathcal{S}^+\times\mathcal{A}\to\mathbb{R}$ 满足 Bellman equation, 则 $Q=Q^\pi$.

**证明**

证明与 value function 的证明基本类似，我们这里略过。

## Bellman Optimal Equation

前面介绍了针对一般策略的 Bellman equation, 特别地，对于我们的目标最优策略，我们也可以而推导出对应的 Bellman equation.

首先我们定义最优策略如下

**Definition**

> 如果策略 $\pi^*$ 满足

$$
V^{\pi^*}(s)\geq V^{\pi}(s), \forall s\in\mathcal{S}^+, \forall \pi
$$

> 则我们称策略 $\pi^*$ 是 **optimal policy**. 对应的 $V^{\pi^*}$ 和 $Q^{\pi^*}$ 分别称之为 **optimal value function** 以及 **optimal Q-function**, 我们简记为 $V^*=V^{\pi^*}$, $Q^*=Q^{\pi^*}$.
> 注意 optimal policy 与状态 $s$ 无关，并且 optimal policy 是不唯一的，但是所有的 optimal policy 对应的 optimal value function 是同一个。
> $V^*$ 与 $Q^*$ 之间存在如下关系

$$
V^*(s) = \max_{a\sim\pi^*} Q^*(s, a)
$$

**证明**

我们先证明左边小于右边，再证明右边小于左边。

$$
V^*(s) =\mathbb{E}_{a_0\sim \pi^*(\cdot\mid s_0)}[Q^*(s,a)]\leq \mathbb{E}_{a_0\sim \pi^*(\cdot\mid s_0)}[\max_{a\sim\pi^*}Q^*(s,a)] =  \max_{a\sim\pi^*} Q^*(s, a)
$$

其次，令 $a^*\in\arg\max_{a}Q^*(s,a)$, 令策略 $\pi'$ 为确定性策略 $\pi'(a^*\mid s)=1$, 则

$$
\max_aQ^*(s,a^*) = Q^{\pi'}(s,a^*)=V^{\pi'}(s)\leq V^*(s)
$$

这样，我们就证明了上面的等式。$\blacksquare$

**Theorem**

> 假设 $\gamma\in(0,1)$, $|\mathcal{S}|<\infty$ $|\mathcal{A}|<\infty$ 以及 $|r|\leq R<\infty, \mathrm{a.s.}$.  那么 optimal value function $V^\pi:\mathcal{S}^+\to\mathbb{R}$ 存在，且满足 **Bellman optimality equation**:

$$
\boxed{
V^*(s) = \max_{a\in\mathcal{A}}\mathbb{E}_{\ (r,s')\sim p(\cdot,\cdot\mid s,a_0)}\left[r+\gamma V^*(s')\mid s, a\right]
}
$$

> 反之，如果存在函数 $V:\mathcal{S}^+\to\mathbb{R}$ 满足 Bellman optimality equation, 则 $V=V^*$, 最终

$$
\pi^*(s) = \arg\max_{a\in\mathcal{A}}\mathbb{E}_{\ (r,s')\sim p(\cdot,\cdot\mid s,a_0)}\left[r+\gamma V^*(s')\mid s, a\right]=\arg\max_{a\in\mathcal{A}} Q^*(s,a)
$$

> 是一个 optimal deterministic policy.

**证明**

我们首先证明存在性，我们定义

$$
(\mathcal{T}^* V)(s) := \max_{a\in\mathcal{A}}\mathbb{E}_{(r,s')\sim p(\cdot,\cdot\mid s,a_0))}\left[r+\gamma V(s')\mid s, a\right]
$$

我们证明 $\mathcal{T}^*V$ 是一个 contraction mapping.

$$
\begin{aligned}
\left|(\mathcal{T}^* V_1)(s)-(\mathcal{T}^* V_2)(s)\right| &= \left|\max_{a\in\mathcal{A}}\mathbb{E}_{(r,s')\sim p(\cdot,\cdot\mid s,a_0))}\left[r+\gamma V_1(s')\mid s, a\right]- \max_{a\in\mathcal{A}}\mathbb{E}_{(r,s')\sim p(\cdot,\cdot\mid s,a_0))}\left[r+\gamma V_2(s')\mid s, a\right]\right|\\
&\leq \max_{a\in\mathcal{A}}\left|\mathbb{E}_{(r,s')\sim p(\cdot,\cdot\mid s,a_0))}\left[r+\gamma V_1(s')\mid s, a\right]- \mathbb{E}_{(r,s')\sim p(\cdot,\cdot\mid s,a_0))}\left[r+\gamma V_2(s')\mid s, a\right]\right|\\
&= \max_{a\in\mathcal{A}}\left|\mathbb{E}_{(r,s')\sim p(\cdot,\cdot\mid s,a_0))}\left[\gamma V_1(s')-V_2(s')\mid s, a\right]- \mathbb{E}_{(r,s')\sim p(\cdot,\cdot\mid s,a_0))}\left[r+\gamma V_2(s')\mid s, a\right]\right|\\
&\leq \gamma  \max_{a\in\mathcal{A}}\left|\mathbb{E}_{a\sim\pi(\cdot\mid s'), (r,s')\sim p(\cdot,\cdot\mid s,a_0)}\left[V_1^{\pi}(s')-V_2^{\pi}(s')\mid s,a\right]\right|\\
&\leq \gamma  \max_{a\in\mathcal{A}}\mathbb{E}_{a\sim\pi(\cdot\mid s'), (r,s')\sim p(\cdot,\cdot\mid s,a_0)}\left[\left|V_1^{\pi}(s')-V_2^{\pi}(s')\right|\mid s,a\right]\\
&\leq \gamma  \max_{a\in\mathcal{A}}\max_{s''\in\mathcal{S}}\left|V_1(s'')-V_2(s'')\right|\\
&= \gamma \left\|V_1-V_2\right\|_\infty
\end{aligned}
$$

这里第一个不等式我们使用了结论

$$
\left|\max_s v(s)-\max_s u(s)\right|\leq \max_s|u(s)-v(s)|
$$

上式对于任意 $s\in\mathcal{S}$ 都成立，因此

$$
\left\|\mathcal{T}^* V_1-\mathcal{T}^* V_2\right\|_\infty \leq \gamma \left\|V_1-V_2\right\|_\infty
$$

由于 $\gamma < 1$, 从而 $\mathcal{T}^*$ 是一个 contraction mapping.

根据不动点定理，$\mathcal{T}^*$ 存在一个不动点，我们将其记为 $V^*$ .

接下来，我们定义 $\pi^*$ 为

$$
\pi^*(s) = \arg\max_{a\in\mathcal{A}}\mathbb{E}_{\ (r,s')\sim p(\cdot,\cdot\mid s,a_0)}\left[r+\gamma V^*(s')\mid s, a\right]
$$

此时，我们有

$$
\begin{aligned}
V^*(s) &=  \max_{a\in\mathcal{A}}\mathbb{E}_{\ (r,s')\sim p(\cdot,\cdot\mid s,a_0)}\left[r+\gamma V^*(s')\mid s, a\right]\\
&= \mathbb{E}_{a\sim\pi^*(s),\ (r,s')\sim p(\cdot,\cdot\mid s,a_0)}\left[r+\gamma V^*(s')\mid s\right]\\
&= \mathbb{E}^{\pi^*}\left[r_0+\gamma V^*(s_1)\mid s_0=s\right]\\
&= \mathbb{E}^{\pi^*}\left[r_0+\gamma \mathbb{E}^{\pi^*}\left[r_1+\gamma V^*(s_2)\mid s_0=s_1\right]\\\mid s_0=s\right]\\
&= \mathbb{E}^{\pi^*}\left[r_0+\gamma \mathbb{E}^{\pi^*}\left[r_1+\gamma V^*(s_2)\mid s_0=s_1, r_0\right]\\\mid s_0=s\right]\\
&= \mathbb{E}^{\pi^*}\left[\mathbb{E}^{\pi^*}\left[r_0+\gamma (r_1+\gamma V^*(s_2))\mid s_0=s_1, r_0\right]\\\mid s_0=s\right]\\
&= \mathbb{E}^{\pi^*}\left[r_0+\gamma r_1+\gamma^2 V^*(s_2)\mid s_0=s\right]\\
&= \mathbb{E}^{\pi^*}\left[r_0+\gamma r_1+\gamma^2r_2+\cdots\mid s_0=s\right]\\
&= V^{\pi^*}(s)
\end{aligned}
$$

即 $V^*=V^{\pi^*}$.

现在我们证明 $\pi^*$ 是最优策略，令 $\pi$ 为任意一个策略，我们有

$$
V^\pi  \leq V^*= \mathcal{T}^*(V^\pi)\leq ( \mathcal{T}^*)^2(V^\pi) \leq\cdots\leq (\mathcal{T}^*)^k(V^\pi) \to V^*=V^{\pi^*}
$$

这里第一个等式是 Bellman equation, 第一个不等式是由于下面的 Lemma,最后的极限使用了不动点定理。因此我们有 $V^\pi\leq V^*$, 从而 $\pi^*$ 是最优策略，且 $V^*$ 是对应的 optimal value function. $\blacksquare$

**Lemma**

> 令 $\pi$ 为一个策略， $\mathcal{T}^\pi$ 和 $\mathcal{T}^*$ 分别是 Bellman operator 和 Bellman optimality operator. 对任意 $V:\mathcal{S}^+\to\mathbb{R}$, 我们有

$$
\mathcal{T}^\pi(V) \leq \mathcal{T}^*(V)
$$

> 进一步，对任意 $U:\mathcal{S}^+\to\mathbb{R}$, $V:\mathcal{S}^+\to\mathbb{R}$, 如果 $U\leq V$, 则

$$
\mathcal{T}^*(U) \leq \mathcal{T}^*(V)
$$

**证明**

$$
\begin{aligned}
\mathcal{T}^\pi(V)(s) &=  \mathbb{E}_{a\sim \pi(\cdot\mid s),\ (r,s')\sim p(\cdot,\cdot\mid s,a_0))}\left[r+\gamma V(s')\mid s\right]\\
&\leq \mathbb{E}_{a\sim \pi(\cdot\mid s)}\left[\max_{a'} \mathbb{E}_{(r,s')\sim p(\cdot,\cdot\mid s,a')}\left[r+\gamma V(s')\mid s\right]\right]\\
&= \max_{a'} \mathbb{E}_{(r,s')\sim p(\cdot,\cdot\mid s,a')}\left[r+\gamma V(s')\mid s\right]\\
&= \mathcal{T}^*(V)
\end{aligned}
$$

其次，令 $a^*\in\arg\max_{a}\mathbb{E}[r+\gamma U(s')]$, 那么

$$
\begin{aligned}
\mathcal{T}^*(U)(s) &=  \mathbb{E}_{(r,s')\sim p(\cdot,\cdot\mid s,a'))}\left[r+\gamma U(s')\mid s\right]\\
&\leq \mathbb{E}_{(r,s')\sim p(\cdot,\cdot\mid s,a'))}\left[r+\gamma V(s')\mid s\right]\\
&\leq \max_a\mathbb{E}_{(r,s')\sim p(\cdot,\cdot\mid s,a'))}\left[r+\gamma V(s')\mid s\right]\\
&= \mathcal{T}^*(V)
\end{aligned}
$$

证毕。 $\blacksquare$

对于 Q-function, 我们也有类似结论

**Theorem**

> 假设 $\gamma\in(0,1)$, $|\mathcal{S}|<\infty$ $|\mathcal{A}|<\infty$ 以及 $|r|\leq R<\infty, \mathrm{a.s.}$.  那么 optimal Q-function $Q^*:\mathcal{S}^+\times\mathcal{A}\to\mathbb{R}$ 存在，且满足 **Bellman optimality equation**:

$$
\boxed{
Q^*(s, a) = \mathbb{E}_{(r,s')\sim p(\cdot,\cdot\mid s,a)}\left[r+\gamma \max_{a'\in\mathcal{A}}Q(s', a')\mid s, a\right]
}
$$

> 反之，如果存在函数 $Q:\mathcal{S}^+\times\mathcal{A}\to\mathbb{R}$ 满足 Bellman optimality equation, 则 $Q=Q^*$.

**证明**

证明与 value function 类似，我们这里略过。

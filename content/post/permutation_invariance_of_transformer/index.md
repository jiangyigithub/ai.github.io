---
title: Notes on attention bias
description: 为什么transformer没有QKV bias
date: 2025-05-22 15:25:07+0800
tags: 
    - transformer
categories:
    - LLM 
math: true
---

## Introduction

我们知道，transformer使用position encoding的一个原因就是，attention layer具有置换不变性，也就是说，我们随机打乱输入token的顺序，并不影响其最终结果 (我们后面会证明，实际上只对key和value具有置换不变性，对query具有置换等变性，也就是改变query的顺序之后，结果的顺序也相应改变)。因此为了让模型学习到正确的上下文知识，我们需要加上position encoding。

已有的工作大部分都在讨论如何构建更好的position encoding，但是鲜有工作探究为什么attention layer具有置换不变性. 因此，本文将从这一点出发，抽丝剥茧探究其内在原因，最后通过数学公式证明原始transformer是如何具有置换不变性的。

## attention layer介绍

原始transformer layer的架构比较简单，其结构具有`attention-LayerNorm-FFN-LayerNorm`的形式。给定输入 $X\in\mathbb{R}^{d\times m}$ 和上下文 $Y\in\mathbb{R}^{d\times n}$. 其中，attention的定义为

$$
\mathrm{Attn}(X, Y, Y) = V\mathrm{softmax}\left(\frac{K^TQ}{\sqrt{d}}\right)\in\mathbb{R}^{d\times m}
$$
其中 $d$是模型的`hidden_size`, $Q=W_QX\in\mathbb{R}^{d\times m}$, $K=W_KY\in\mathbb{R}^{d\times n}$, $V=W_VY\in\mathbb{R}^{d\times n}$, $W_Q, W_K, W_V\in\mathbb{R}^{d\times d}$ 分别是QKV projection layer的参数.

LayerNorm的定义为：

$$
\mathrm{LayerNorm}(x) = \frac{x-\mathbb{E}[x]}{\sqrt{\mathrm{var}[x]+\epsilon}}\odot \gamma + \beta
$$
其中 $\epsilon>0$是一个超参数， $\gamma, \beta\in\mathbb{R}^d$ 是可学习的参数.

FFN的定义为：

$$
\mathrm{FFN}(x) = W_2\max(0, W_1x+b_1)+b_2
$$
其中 $W_1\in\mathbb{R}^{d_{ff}\times d}$, $W_2\in\mathbb{R}^{d\times d_{ff}}$, $b_1\in\mathbb{R}^{d_{ff}}$, $b_2\in\mathbb{R}^d$ 是可学习的参数。

最后，一个attention layer的的结构可以表达为：

$$
X = X + \mathrm{LayerNorm}(\mathrm{Attn}(X, Y, Y))\\
X = X + \mathrm{LayerNorm}(\mathrm{FFN}(X))\\
$$

### 置换不变性的定义

置换不变性(permutation invariant)的定义：假设 $f:\mathbb{R}^n\to\mathbb{R}^n$，如果

$$
f(\sigma(\bm{x})) = (f(\bm{x}))
$$

则我们说 $f$是置换不变的. 这里 $\sigma:\mathbb{R}^n\to\mathbb{R}^n$ 是一个置换函数 (permutation function). 当输入的是一个矩阵时，我们默认置换其列，即对 $X=[X_1,\dots,X_n]\in\mathbb{R}^{d\times n}$, 我们有 $\sigma(X)=[X_{\sigma_1},\dots, X_{\sigma_n}]=Y\Pi $, 其中 $\Pi\in\mathbb{R}^{n\times n}\in \{0,1\}^{n\times n}$ 是一个置换矩阵 (permutation matrix)。

置换等变性 (permutation equivariant)的定义：假设 $f:\mathbb{R}^n\to\mathbb{R}^n$，如果

$$
f(\sigma(\bm{x})) = \sigma(f(\bm{x}))
$$
则我们说 $f$是置换等变的.

## attention的置换不变性与置换等变性

我们首先证明attention 对于key和value是置换不变的，即

$$
\boxed{\mathrm{Attn}(X, \sigma(Y),\sigma(Y)) = \mathrm{Attn}(X, Y, Y)}
$$

**证明**: 我们直接计算即可得到：

$$
\begin{aligned}
    \mathrm{Attn}(X, \sigma(Y),\sigma(Y)) &= V\Pi\mathrm{softmax}\left(\frac{(K\Pi)^TQ}{\sqrt{d}}\right)\\
    &=V\Pi\mathrm{softmax}\left(\frac{\Pi^TK^TQ}{\sqrt{d}}\right)\\
\end{aligned}
$$
由于softmax是按列计算的，置换只是改变了元素的顺序，因此我们自然有

$$
\mathrm{Attn}(X, \sigma(Y),\sigma(Y)) = V\Pi\Pi^T\mathrm{softmax}\left(\frac{K^TQ}{\sqrt{d}}\right)=V\mathrm{softmax}\left(\frac{K^TQ}{\sqrt{d}}\right)=\mathrm{Attn}(X, Y, Y)
$$
这里我们使用了性质 $\Pi\Pi^T=\mathbf{I}$.

接下来我们证明，attention对于query是置换等变的，即

$$
\boxed{\mathrm{Attn}(\sigma(X), Y, Y) = \sigma(\mathrm{Attn}(X,Y,Y))}
$$

**证明**:
$$
\begin{aligned}
    \mathrm{Attn}(\sigma(X), Y, Y) &= V\mathrm{softmax}\left(\frac{K^TQ\Pi}{\sqrt{d}}\right)\\
    &= V\mathrm{softmax}\left(\frac{K^TQ\Pi}{\sqrt{d}}\right)\\
    &= V\mathrm{softmax}\left(\frac{K^TQ}{\sqrt{d}}\right)\Pi\\
    &= \mathrm{Attn}(X,Y,Y)\Pi\\
    &= \sigma(\mathrm{Attn}(X,Y,Y))
\end{aligned}
$$

从以上的证明可以看到，attention layer对于key和value具有置换不变性，也就是说，我们改变文字顺序不影响最终的输出结果。
但是，我们发现，尽管我们证明了attention具有置换不变性，我们却忽略了一件事：那就是我们计算query, key和value的时候，没有加上bias! 为什么bias如此重要呢？这是因为，$W\sigma(x) = \sigma(Wx)$, 但是 $W\sigma(X)\neq \sigma(Wx+b)$.
因此，我们就会思考，难道是transformer实际上可以通过增加bias的方式来让模型学习到上下文知识？事实上并非如此，我们将要通过分析表明，我们计算query, key和value时，增加的query bias和key bias会被softmax操作给消除掉，而key bias则会被LayerNorm消除掉。因此，我们加与加bias，对attention的置换不变性没有任何影响。

## Bias对attention layer的影响

接下来，我们考虑在计算query, key和value时加入bias。为了简化，我们只考虑query为一个向量的情况，即 $X=\bm{x}\in\mathbb{R}^d$, 我们计算query, key和value如下：

$$
\bm{q} = W_Q\bm{x}+\bm{b}_Q\in\mathbb{R}^{d}\\
K = W_KY + \bm{b}_K\mathbf{1}^T\in\mathbb{R}^{d\times n}\\
V = W_VY + \bm{b}_V\mathbf{1}^T\in\mathbb{R}^{d\times n}
$$

这里 $\mathbf{1}^T\in\mathbb{R}^{n}$. 我们这里简化了scaling的操作，因为其不对结果产生影响。

> 注：以下证明参考了【参考文献2】

我们首先展开attention中的 $V$:

$$
\begin{aligned}
    \mathrm{Attn}(\bm{x}, Y, Y) &= V\mathrm{softmax}\left(K^T\bm{q}\right)\\
    &= \left(W_VY + \bm{b}_V\mathbb{1}^T\right)\mathrm{softmax}\left(K^T\bm{q}\right)\\
    &= W_VY\mathrm{softmax}\left(K^T\bm{q}\right) + \bm{b}_V\mathbb{1}^T \mathrm{softmax}\left(K^T\bm{q}\right)
\end{aligned}
$$
由于 $\mathrm{softmax}\left(K^T\bm{q}\right)\in\mathbb{R}^{n}$的列求和为$1$, 因此，$\mathbb{1}^T\mathrm{softmax}\left(K^T\bm{q}\right)=1$, 我们有

$$
\mathrm{Attn}(\bm{x}, Y, Y) = W_VY\mathrm{softmax}\left(K^T\bm{q}\right) + \bm{b}_V
$$

接下来，我们展开 $K$:

$$
\begin{aligned}
\mathrm{Attn}(\bm{x}, Y, Y) &= W_VY\mathrm{softmax}\left(K^T\bm{q}\right) + \bm{b}_V\\
&= W_VY\mathrm{softmax}\left((W_KY + \bm{b}_K\mathbf{1}^T)^T\bm{q}\right) + \bm{b}_V\\
&= W_VY\mathrm{softmax}\left(Y^TW_K^Tq + \mathbf{1}\bm{b}_K^T\bm{q}\right) + \bm{b}_V\\
\end{aligned}
$$

这里，我们需要用到softmax函数的平移不变性，即 $\mathrm{softmax}(\bm{x}+\delta\mathbf{1})=\mathrm{softmax}(\bm{x})$, 这里 $\delta\in\mathbb{R}$ 是一个常数，证明起来很简单：
$$
\mathrm{softmax}(\bm{x}+\delta)_i = \frac{e^{x_i+\delta}}{\sum_{j}e^{x_j+\delta}} = \frac{e^{x_i} * e^{\delta}}{\sum_{j}e^{x_j} * e^{\delta}} = \mathrm{softmax}(\bm{x})_i
$$
而这里 $\bm{b}_K^T\bm{q}\in\mathbb{R}$，因此我们可以将这一项给去掉，我们得到：

$$
\mathrm{Attn}(\bm{x}, Y, Y) = W_VY\mathrm{softmax}\left(Y^TW_K^T\bm{q}\right) + \bm{b}_V
$$

接下来，我们展开 $\bm{q}$,
$$
\boxed{
\begin{aligned}
\mathrm{Attn}(\bm{x}, Y, Y) &= W_VY\mathrm{softmax}\left(Y^TW_K^T\bm{q}\right) + \bm{b}_V\\
&= W_VY\mathrm{softmax}\left(Y^TW_K^T(W_Q\bm{x}+\bm{b}_Q)\right) + \bm{b}_V\\
\end{aligned}}
$$

因此，我们最终的结论为： **key bias对attention输出没有任何贡献，query bias和key bias会影响结果。**

到这里，看了参考文献3，我本以为可以进一步简化。但实际上并不行。参考文献3关于“transformer block is equivariant"的结果是错的，因为在attention layer之后还有一个LayerNorm，而LayerNorm不是置换不变的，这也是LayerNorm和BatchNorm之间的区别。也就是*如果我们在`nn.Linear`后加一个BatchNorm，那么`nn.Linear`的bias是无效的，反之如果是LayerNorm的话，则bias是有效的*.

## 为什么没有bias

实际上这个问题并没有定论。特别是加入position encoding之后，就更难探究bias对最终结果的影响了。但是，我认为一个原因就是bias其实就是某种先验知识，假设输入满足高斯分布，那么我们有

$$
\mathbb{E}[W\bm{x}+b] = b
$$

加上先验知识后，当训练数据出现distribution shift之后，模型在训练过程中可能就会不稳定(PaLM). 而后来将LayerNorm替换为RMSNorm，使用RoPE而不是其他的additive position encoding, 我认为也是避免模型学习到先验知识，从而影响其泛化性。在未来，我认为transformer里应该是没有bias的，尽管这样效果可能会差一些，但是其稳定性更好，泛化性应该也会更好。

## 结论

在本文中，我们分析了attention的性质，我们发现，在原始transformer架构中，attention对于key和value有置换不变性，对于query有置换等变性。然后，我们给出了一些猜测，也就是bias会让模型产生先验知识，而这种先验知识很可能会影响训练的稳定性和模型的泛化性。

## 参考文献

- [Attention is All you Need](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
- [Role of Bias Terms in Dot-Product Attention](https://arxiv.org/abs/2302.08626)
- [Are Transformers universal approximators of sequence-to-sequence functions?](https://openreview.net/forum?id=ByxRM0Ntvr)

## 附录

下面是测试上面结论的python代码

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# 设置随机种子，确保可复现性
torch.manual_seed(42)

# 输入参数
batch_size = 1
seq_len = 16
embed_dim = 1024  # 嵌入维度
num_heads = 32  # 多头注意力头数
head_dim = embed_dim // num_heads

# 输入张量 (batch_size, seq_len, embed_dim)
x = torch.randn(batch_size, seq_len, embed_dim)


# 有 bias 的 QKV 线性层
class Attention(nn.Module):
    def __init__(self, embed_dim, q_bias=False, k_bias=False, v_bias=False):
        super().__init__()
        self.q = nn.Linear(embed_dim, embed_dim, bias=q_bias)
        self.k = nn.Linear(embed_dim, embed_dim, bias=k_bias)
        self.v = nn.Linear(embed_dim, embed_dim, bias=v_bias)

    def forward(self, x):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, num_heads, head_dim).transpose(1, 2)
        k = self.k(x).reshape(B, N, num_heads, head_dim).transpose(1, 2)
        v = self.v(x).reshape(B, N, num_heads, head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * (1.0 / (head_dim**0.5))
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return x, attn


# 初始化模型
model_no_bias = Attention(embed_dim, q_bias=False, k_bias=False, v_bias=False)
model_with_bias = Attention(embed_dim, q_bias=False, k_bias=True, v_bias=False)

model_with_bias.q.weight.data = model_no_bias.q.weight.data
model_with_bias.k.weight.data = model_no_bias.k.weight.data
model_with_bias.v.weight.data = model_no_bias.v.weight.data

# 推理
out_no_bias, attn_no_bias = model_no_bias(x)
out_with_bias, attn_with_bias = model_with_bias(x)


# 比较差异
diff_output = torch.abs(out_no_bias - out_with_bias).mean()
diff_variance = torch.abs(out_no_bias - out_with_bias).var()
diff_attn = torch.abs(attn_no_bias - attn_with_bias).mean()

print("\nMean difference in output:", diff_output.item())
print("Mean difference in variance:", diff_variance.item())
print("Mean difference in attention weights:", diff_attn.item())

# Mean difference in output: 1.2734082233123445e-08
# Mean difference in variance: 1.7173628739783402e-16
# Mean difference in attention weights: 3.949708116124384e-09
```

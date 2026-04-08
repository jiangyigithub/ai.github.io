---
title: Notes on MQA
description: Google 在 2019 年提出了 multi-query attention (MQA), 用于解决 MQA 内存带宽瓶颈问题。
date: 2025-08-07 18:06:37+0800
lastmod: 2025-08-07 18:06:37+0800
math: true
tags: 
    - attention
categories:
    - LLM 
---


Google 在 2019 年提出了 multi-query attention (MQA), 用于解决 MQA 内存带宽瓶颈问题。

## Method

### Background

对于 multi-head attention, 我们假设其 hidden size 为 $d$, 有 $h$ 个 heads, 每个 head 的 size 为 $d_h=d/h$, 输入 sequence 长度为 $n$, batch size 为 $d$. 则总的 arithmetic operations 为 $O(bnd^2)$. 总的内存访问量为 $O(bnd + bhn^2+d^2)$, 第一项是 $Q,K,V$ 的内存占用（$Q,K,V$ 分别是 query, key 和 value layer 的输出），第二项是 attention score 的占用，第三项是 query, key 和 value layer 的权重。

因此，其 **Memory Access Ratio** (MAR), 也就是内存访问量 与 arithmetic operations 之比为

$$
O\left(\frac1k + \frac{1}{bn}\right)
$$

对于现代的 GPU 来说，其一般算力比较强，但是内存访问带宽相对较慢，因此我们希望 MAR 越低越好，以充分发挥 GPU 的算力。

### MHA Analysis

在训练的时候，由于我们知道 ground truth sequence, 因此我们可以并行计算。但是在 inference 的时候，我们只能 token-by-token 进行计算，因此我们分析一下 token-by-token 场景下的 MAR

我们整体的 arithmetic operations 还是 $O(bnd^2)$.

但是，现在我们要调用 $n$ 次 multi-head attention, 因此我们总的内存访问量为 $O(bn^2d + nd^2)$,  第一项是 $K$ 和 $V$ , 第二项是 query, key 和 value layer 的权重。

这种情况下，MAR 就变成了

$$
O\left(\frac{n}{d} + \frac{1}{b}\right)
$$

当 $n\approx d$ 或者 $b\approx 1$ 时，MAR 就非常接近于 1，意味着内存带宽成了一个主要的瓶颈。为了解决这个问题，我们有两种做法：

1. 提升 batch size $b$, 也就是同时 inference 多次
2. 降低 $K$ 和 $V$ 的大小

### MQA

MQA 的做法就是第二种，也就是降低 $K$ 和 $V$ 的大小，但是 $K,V$ 分别是 key 和 value layer 的输出，要降低输出大小，我们就必须改变 key 和 value layer 的 size。基于这个考虑，作者在所有的 head 上共享了一个 key 和 value layer，也就是说，原来

```python
self.k_proj = nn.Linear(hidden_size, num_heads * head_dim) # (d, n*d_h)
self.v_proj = nn.Linear(hidden_size, num_heads * head_dim) # (d, n*d_h)
```

现在在 MQA 里，其变成了

```python
self.k_proj = nn.Linear(hidden_size, head_dim) # (d, n*d_h)
self.v_proj = nn.Linear(hidden_size, head_dim) # (d, n*d_h)
```

### MQA Analysis

我们还是在 token-by-token 的场景下进行分析。

我们整体的 arithmetic operations 还是 $O(bnd^2)$.

调用 $n$ 次 multi-query attention 的总的内存访问量为 $O(bnd +bn^2d_h+ nd^2)$,  第一项是 $q$  , 第二项是 $K$ 和 $V$ , 第三项是是 query, key 和 value layer 的权重。

此时，MAR 变成了

$$
O\left(\frac{1}{d} + \frac{n}{dh}+\frac{1}{b}\right)
$$

现在，我们就将 $n/d$ 这一项给降低了 $h$ 倍。如果我们的 batch size 足够大的话，理论上 MQA 应该能极大提高整体的计算效率。

## Conclusion

MQA 为了追求极致的内存带宽占用，选择使用单一的 key 和 value, 来极大提高 inference 的 decoding 效率，但是后来在 GQA 中验证发现，MQA 虽然非常高效，但是其表现比较差，这也是后来没有得以应用的原因。

## References

- [Arxiv](https://arxiv.org/pdf/1911.02150)

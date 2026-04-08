---
title: Mixstral 8x7B
description: Mistral 在 24 年 1 月提出了 Mistral 8x7B, 一个 MoE 大语言模型，模型包括 8 个专家，激活 2 个专家，总参数量为 47B, 激活参数量为 13B.
date: 2025-11-01 15:32:30+0800
lastmod: 2025-11-01 15:32:30+0800
math: true
tags: 
    - Mixtral
    - MoE
categories:
    - LLM 
---

## Introduction

作者在本文中提出了 Mixtral 8x7B, 一个 MoE 模型，模型上下文为 32K. 作者还对模型进行 finetune 得到了 Mixtral 8x7B-Instruct, finetuning 包含 SFT 和 [DPO](https://maosong.website/p/notes-on-dpo/) 两个阶段。

## Method

模型架构与 [Mistral-7B](https://maosong.website/p/mixstral-7b/) 基本相同，参数如下表所示

| Parameter       | Value |
| --------------- | ----- |
| `dim`           | 4096  |
| `n_layers`      | 32    |
| `head_dim`      | 128   |
| `hidden_dim`    | 14336 |
| `n_heads`       | 32    |
| `n_kv_heads`    | 8     |
| `window_size`   | 4096  |
| `context_len`   | 32768 |
| `vocab_size`    | 32000 |
| `num_experts`   | 8     |
| `top_k_experts` | 2     |

MoE 的架构与 [GShard](https://maosong.website/p/gshard/) 基本一致

## Results

作者探究了专家的 specialization, 结果有三点发现：

1. 不同专家对于不同 domain 的数据并没有出现 specialization
2. 在 math domain 上，专家的分布有一个明显的区别。
3. 连续的 token 往往会被分配到同一个专家上

## Conclusion

作者在本文提出了 Mistral 8x7B, 一个 MoE 大语言模型

## References

- [Arxiv](http://arxiv.org/abs/2401.04088)

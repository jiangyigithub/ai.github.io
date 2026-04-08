---
title: Notes on Qwen1.5
description: Qwen在24年1月份发布了Qwen1.5，包含 0.5B, 1.8B, 4B, 7B, 14B, 32B, 72B, 以及 110B 6个size，还有一个MoE模型。
date: 2025-07-03 17:37:39+0800
lastmod: 2025-07-12 10:35:10+0800
tags: 
    - Qwen
categories:
    - LLM 
---


Qwen 在 24 年 1 月份发布了 Qwen1.5，包含 0.5B, 1.8B, 4B, 7B, 14B, 32B, 72B, 以及 110B 6 个 size，还有一个 MoE 模型。

## 介绍

Qwen1.5 的主要特点：

1. 支持 12 中语言
2. 统一支持 32768 tokens 上下文长度 。
3. 提供 量化版本 （Int4、Int8、AWQ、GGUF）以适应低资源环境或部署需求。

训练过程使用了 [DPO](https://maosong.website/p/notes-on-dpo/) 以及 PPO 来进行对齐

## Qwen1.5-MoE

Qwen1.5-MoE 的激活参数为 2.7B，一共包含 64 个专家，其中激活 4 个专家，共享 4 个专家

相比于 Qwen1.5-7B，去训练的 FLOPS 降低了 75%，inference 的速度提高了 174%

Qwen1.5-MoE 采用了改进的 MoE 架构，主要优化包括：

- **细粒度专家（Fine-grained experts）** ：通过将 FFN 层划分为多个片段，构建更多专家而不增加参数总量。
- **初始化策略（Upcycling）** ：基于 Qwen-1.8B 初始化模型，并引入随机性以加速收敛。
- **路由机制（Routing Mechanism）** ：在每个 MoE 层中使用 64 个专家，其中 4 个共享专家始终激活，60 个路由专家中有 4 个被激活，提高了灵活性和效率。

## 效率对比

作者对比了 throughput (requests processed per second) 以及 tokens per second (TPS):

| Model                  | Throughput | TPS     |
| ---------------------- | ---------- | ------- |
| Qwen1.5-7B-Chat        | 1.15       | 2298.89 |
| Qwen1.5-MoE-A2.7B-Chat | 2.01       | 4010.27 |

## References

- [Qwen1.5 MoE](https://qwenlm.github.io/blog/qwen-moe/)
- [Qwen 1.5](https://qwenlm.github.io/blog/qwen1.5/)

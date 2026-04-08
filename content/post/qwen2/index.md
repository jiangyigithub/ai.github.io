---
title: Notes on Qwen2
description: Qwen2 技术报告总结
date: 2025-07-12 10:36:43+0800
lastmod: 2025-07-12 10:36:43+0800
tags: 
    - Qwen
categories:
    - LLM 
---


## Introduction

2024 年 9 月 Qwen 发布了 Qwen2 系列技术报告，Qwen2 系列包括 4 个 dense 模型（0.5B, 1.5B, 7B, 72B）和一个 MoE 模型（总参数 57B，激活参数 14B），作者主要在架构，数据和长上下文上进行了改进。

## Method

### Model

对于 dense 模型，Qwen2 在 [Qwen-LLM](https://maosong.website/p/notes-on-qwen-llm/) 的基础上做了如下改动：

1. 使用 [Group Query Attention (GQA)](https://maosong.website/p/notes-on-gqa/) 替换 MHA，来优化 KV cache，提高 throughput
2. 使用 [Dual Chunk Attention](https://maosong.website/p/dual-chunk-attention/) 和 [YARN](https://maosong.website/p/notes-on-yarn/) 来提高模型上下文长度和训练效率

其余与 Qwen 一致，包括 SwiGLU，RoPE，RMSNorm 和 pre-normalization

对于 MoE 模型，Qwen2-MoE 基于 [Qwen1.5](https://maosong.website/p/dual-chunk-attention/) 进行了改进，主要是 3 点：

1. 作者使用了更细粒度的专家个数，作者认为细粒度的专家可以提供更丰富的 combination，这一点与 [olmoe](https://maosong.website/p/notes-on-olmoe/) 的结论相同
2. 与 DeepSeek-MoE 一样，作者使用了共享专家和路由专家
3. 作者使用了类似 upcycling 的方法来初始化模型。假设一共有 $n$ 个专家，每个专家的维度为 $h_E$, 原始 dense 模型的维度为 $h_{FFN}$, 那么我们会把 dense 模型的参数复制 $[nh_E/h_{FFN}]$ 次，这样就可以扩展到任意个数的 MoE 模型上。作者还对参数进行 shuffle，来提高 diversity。最后，作者还对 50% 的参数进行随机初始化，来提高模型的 capacity。

模型配置如下

| Configuration       | 0.5B    | 1.5B    | 7B      | 72B     | 57B-A14B |
|---------------------|---------|---------|---------|---------|----------|
| Hidden Size         | 896     | 1,536   | 3,584   | 8,192   | 3,584    |
| # Layers            | 24      | 28      | 28      | 80      | 28       |
| # Query Heads       | 14      | 12      | 28      | 64      | 28       |
| # KV Heads          | 2       | 2       | 4       | 8       | 4        |
| Head Size           | 64      | 128     | 128     | 128     | 128      |
| Intermediate Size   | 4,864   | 8,960   | 18,944  | 29,568  | 2,560    |
| # Routed Experts    | -       | -       | -       | -       | 64       |
| # Activated Experts | -       | -       | -       | -       | 8        |
| # Shared Experts    | -       | -       | -       | -       | 8        |
| Embedding Tying     | True    | True    | False   | False   | False    |
| Vocabulary Size     | 151,646 | 151,646 | 151,646 | 151,646 | 151,646  |
| # Trained Tokens    | 12T     | 7T      | 7T      | 7T      | 4.5T     |

### Pre-training

预训练阶段的数据基于 Qwen 和 Qwen1.5，数据处理策略如下：

1. 使用基于 heuristic 和 model-based 方法来过滤掉低质量的数据
2. 加入了 code， math 和 multilingual 的数据
3. 平衡了各个类别的数据分布

初始数据包括 12T token，经过过滤得到 7T token。作者发现，使用 12T token 进行训练，模型的表现不如使用 7B token 训练得到的模型效果好。因此除了 0.5B 的模型，其他模型使用的都是 7T 的 token

对于 MoE 模型，作者使用了额外的 4.5T token 来进行预训练。

在训练过程中，作者还加入了 multi-task instruction 数据，来提高模型的上下文学习能力和指令跟随能力。

作者还将 Qwen2 模型系列的上下文长度从 4096 扩展到 32768，扩展过程中作了三个改动：

1. 加入了更多高质量的长上下文数据
2. 将 RoPE 的 frequency 从 10,000 提升到了 1,000,000
3. 使用了 [YARN](https://maosong.website/p/notes-on-yarn/) 来扩展上下文长度
4. 使用了 [Dual Chunk Attention](https://maosong.website/p/dual-chunk-attention/) 来优化 attention 的计算

### Post-training

post-training 包括 SFT 和 RLHF 两个阶段

数据包括 SFT 数据和 RLHF 使用的偏好数据

数据标注过程有：

1. 使用 InsTag 对数据进行打标
2. 选取高质量的 instruction
3. 构建了一个 self-evolution 策略，来扩展 instruction 数据
4. 请人类来标注数据

作者还合成了一些数据，合成数据的过程如下：

1. rejection sampling：对 LLM 进行多次采样，然后保留结论正确的数据作为 SFT 数据，以正确和错误的数据对作为偏好数据
2. Execution feedback：对于代码任务，使用 Python 来验证答案的正确性
3. Data Repurposing：对于写作类任务，以文档为输入，让 LLM 生成对应的 instruction
4. Constitutional Feeback：基于预设的 principle 来生成回答

最终，SFT 数据包括 500, 000 条样本

RLHF 的训练包括 offline stage 和 online stage，offline stage 就是用收集到的偏好数据。在 online stage，作者使用 reward model 来给输出的回答进行打分，然后再使用 DPO 进行训练。

与 Qwen 不同，Qwen2 中作者使用了 Online Merging Optimizer 来解决因为 alignment 导致的性能降低

## Conclusion

本文提出了 Qwen2 系列，在 Qwen2 中，首次使用了 GQA 代替 MHA，Qwen2 在上下文上做出了初步探索

## References

- [Qwen2 tech report](https://arxiv.org/abs/2407.10671)

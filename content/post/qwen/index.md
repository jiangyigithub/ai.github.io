---
title: Notes on Qwen-LLM
description: Qwen技术报告总结
date: 2025-07-03 10:47:27+0800
lastmod: 2025-07-12 10:32:03+0800
tags: 
    - Qwen
categories:
    - LLM 
math: true
---


Qwen 在 23 年 9 月份发布了 Qwen 系列大语言模型，包括 1.8B， 7B，14B 三个 size，训练过程使用了 3T token. 作者还基于 Qwen，构建了 Code-Qwen-Chat，Math-Qwen-Chat 等系列领域大语言模型。

## Pre-training

### Data

数据一共使用了 **3T token**，主要是 public web documents, encyclopedia, books, codes, etc，覆盖了中文和英文两种语言

数据处理：

1. 语言识别
2. 去重，包括 MinHash 和 LSH 算法
3. 质量过滤，包括基于规则和和基于 ML 的方法
4. 上采样，特定数据会进行上采样
5. 加入指令数据，提高模型的 zero-shot 和 few-shot 表现

### Tokenization

BPE tokenizer，最终的 tokenizer 大小为 152K

### Architecture

模型架构基于 LLaMA， 改动：

1. tie embdding: input embdding 和 output embdding 使用的权重相同
2. position encoding:RoPE, inverse frequency 的精度为 FP32
3. bias: 取消了大部分的 bias，增加了 QKV bias，来提高模型的外推能力
4. Pre-Norm & RMSNorm
5. Activation function: SwiGLU

### Training

- 上下文长度：2048
- attention：[flash attention](https://maosong.website/p/notes-on-flashattention/)
- optimizer：AdamW， $\beta_1=0.9$, $\beta_2=0.95$, $\epsilon=10^{-8}$.
- data type: BF16

### Context Extention

使用了三个技巧：

1. NTK-aware position interpolation
2. log-N scaling
3. window attention

后续前两个统一成了 [YARN](https://maosong.website/p/notes-on-yarn/).

observation: lower layer 对上下文长度扩展更敏感, 因此作者动态调整了 window size

## Post-training

包括 SFT 和 RLHF 两个阶段

### SFT

data： 使用了 ChatML 格式

### RLHF

PPO 算法

reward model 构建：基于 Qwen-base model

RL 训练：先更新 value model 50 steps

发现：top-p 设置为 0.9 比设置为 1.0 更好

### Tool-use and Agent

作者使用了 self-instruct 来进行 SFT，基于 ReAct 构建数据，数据包括 2000 条高质量数据

## Specialization

### Code-Qwen

code-qwen 基于 qwen continue Pretraining 得到，然后基于 code-qwen 进行 sft 得到 code-qwen-chat，包括 7B 和 14B 两个 size

### Math-Qwen

基于 qwen 直接 SFT 得到，包括 7B 和 14B 两个 size

## Conclusion

作者在本文中介绍了 Qwen 系列大语言模型，模型使用了 3T token，作者介绍了训练的细节以及如何扩展到领域大语言模型 Code-Qwen 和 Math-Qwen

## References

- [Length exploration](https://spaces.ac.cn/archives/9444)
- [Arxiv](https://arxiv.org/abs/2309.16609)

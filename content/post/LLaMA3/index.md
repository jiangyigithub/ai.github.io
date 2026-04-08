---
title: Notes on Llama3
description: An brief introduction to Llama3
date: 2024-04-22 16:22:19+0800
tags: 
    - Llama
categories:
    - LLM 
---

Meta released Llama3 at April 18, which is evaluated on several benchmarks and achieves the SOTA on open-sourced LLMs.

# Introduction

## Instruct model performance

The performance of Llama3 8B compared with Gemma and Mistral:

| Model                  | Llama3 8B   | Gemma 7B  - It   |  Mistral &B Instruct  |
| --------               | --------    | ------           |  ------               |
| *MMLU* (5 shot)        | **68.4**    | 53.3             |  58.4                 |
| *GPQA* (0 shot)        | **34.2**    | 21.4             |  26.3                 |
| *HumanEval* (0 shot)   | **62.2**    | 30.5             |  36.6                 |
| *GSM-8K* (8 shot, CoT) | **79.6**    | 30.6             |  39.9                 |
| *MATH* (4 shot, CoT)   |  **30.0**   | 12.2             |  11.0                 |

performance of Llama3 70B compared with Gemini Pro 1.5 and Claude Sonnet:

| Model                  | Llama3 70B   | Gemini Pro 1.5 (Published) |  Claude 3 Sonnet (Published) |
| --------               | --------    | ------           |  ------                         |
| *MMLU* (5 shot)        | **82.0**    | 81.9                       |  79.0                 |
| *GPQA* (0 shot)        | 39.5        | **41.5** (CoT)             |  38.5 (CoT)           |
| *HumanEval* (0 shot)   | **81.7**    | 71.9                       |  73.0                 |
| *GSM-8K* (8 shot, CoT) | **93.0**    | 91.7 (11 shot)             |  92.3 (0 shot)        |
| *MATH* (4 shot, CoT)   |  50.4       | **58.5**   (Minerva prompt)|  40.5                 |

## Pre-trained model performance

The performance of Llama3 8B compared with Gemma and Mistral:

| Model  | Llama3 8B   | Gemma 7B (Published)   | Gemma 7B (Measured)  |  Mistral 7B (Published) | Mistral 7B (Measured)  |
| --------               | --------    | ------           |  ------               | ------    | ------           |
| *MMLU* (5 shot)        | **66.6**    | 64.3            |  64.4                 | 62.5       |    63.9                |
| *AGIEval English* (3-5 shot)        | **45.9**    | 41.7            |  44.9                 | -       |    44.0               |
| *BIG-Bench Hard* (3 shot, CoT)        | **61.1**    | 55.1          |  59.0                | -       |    56.0                |
| *ARC-Challenge* (25 shot)        | 78.6    | 53.2(0 shot)      |  **79.1**     | 78.1       |   78.7     |
| *DROP* (3 shot, F1)        | **58.4**    | -           |  56.3       | -      |    54.4        |

performance of Llama3 70B compared with Gemini Pro 1.5 and Claude Sonnet:

| Model                  | Llama3 70B   | Gemini Pro 1.0 (Published) |  Mixtral 8 $\times$ 22B (Measured) |
| --------               | --------    | ------                      |  ------                         |
| *MMLU* (5-shot)        | **79.5**    | 71.8                        |  77.7                 |
| *AGIEval English* (3-5 shot)       | **63.0**    | -                           |  61.2                 |
| *BIG-Bench Hard* (3 shot, CoT)   | **81.3**    | 75.0                        | 79.2                  |
| *ARC-Challenge* (25 shot) | **93.0**    | -                           |  90.7                 |
| *DROP* (3 shot, F1)    |  **79.7**   | 74.1 (variable shot)        |  77.6                 |

# Model Architecture

Several improvements are made on Llama3 compared to llama2:

1. Llama3 uses a tokenizer with a vocabulary of 128K tokens.
2. Llama3 adopts [grouped query attention (GQA)](https://maosong.website/p/notes-on-gqa/) across both the 8B and 70B sizes.
3. Llama3 uses to context window of size 8192 tokens

# Traning

Llama3 uses 15T tokens for pre-training. Compares to Llama2, it is seven times larger and includes four times more code.

5% data of the training dataset are non-English to support multi-lingual use cases.

Data processing includes:

1. Heuristic filters
2. NSFW filters
3. Semantic deduplication approaches
4. Text classifiers to predict data quality. Llama2 is used to generate training data for the text classifiers.

Data mixing strategy is explored to improve the performance of Llama3.

# Scaling up pretraining

Llama3 developed a series of scaling laws for downstream benchmark evaluations.

Scaling laws help:

1. Select an optimal data mix and to make informed decisions on how to best use training compute.
2. Scaling laws allow Llama3 to predict the performance of the largest models on key tasks without training the models.

The authors finds our that the performance of the model continues to improve log-linearly as the training tokens increase. It is seen that  Larger models can match the performance of these smaller models with less training compute, but smaller models are generally preferred because they are much more efficient during inference.

The authors combine three types of parallelization:

1. Data parallelization
2. Model parallelization
3. Pipeline parallelization

# Instruction fine-tuning

The fine-tuning of Llama3 contains:

1. Supervised fine-tuning
2. Rejection sampling
3. Proximal Policy Optimization
4. Direct Preference Optimization

Learning from perference rankings via PPO and [DPO](https://maosong.website/p/notes-on-dpo/) also greatly improved the performance of LLma3 on reasoning and coding tasks. Since perference ranking helps the model to select answer when it is in a dilemma.

# Reference

- [Llam3 blog](https://ai.meta.com/blog/meta-llama-3/)  
- [Evaluation details](https://github.com/meta-llama/llama3/blob/main/eval_details.md)
- [Model Card](https://github.com/meta-llama/llama3/blob/main/MODEL_CARD.md)

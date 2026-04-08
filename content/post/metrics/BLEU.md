---
title: BLEU (Bilingual Evaluation Understudy)
date: 2024-04-25 22:46:53+0800
description: The metric that evaluates the quality of the translation
tags: 
    - Metric
categories:
    - LLM 
    - NLP
math: true
---

BLEU (Bilingual Evaluation Understudy) is a widely used metric that evaluates the quality of the translated text with respect to the reference translations.

# Introduction

The formula of BLEU is defined as follows:

$$ \mathrm{BLEU}_ {w_n}(\hat{S}, S) = \mathrm{BP} \cdot \exp\left(\sum_{n=1}^Nw_n\log p_n(\hat{S}, S) \right) $$

where

1. $\mathrm{BP}$ represents the Brevity Penalty to penalize the translations that are shorter than the reference translations.
2. $p_n$ represents the modified $n$-gram precision score
3. $w_n$ represents the weight for $p_n$, it satisfies $0\leq w_n\leq1$ and $\sum_{n=1}^Nw_n=1$.

Now we interprets three parts in detail.

# Interpretation

## Definitions

Given any string $y=y_1y_2\cdots y_K$, where $y_i,i\in{1,\dots,K}$ are characters and an integer $n\geq1$, we define the **set of $n$-gram** to be

$$ G_n(y) = \{ y_1\cdots y_n, y_2\cdots y_{n+1}, \dots, y_{K-n+1}\cdots y_K\} $$

NOTE that this is a set with unique elements, for example, $G_2(abab)=\{ab, ba\}$.

Given any two strings $s$ and $y$, we define **substring count** $C(s,y)$ to tbe the number of appearances of $s$ as a substring of $y$. For example, $C(ab, abcbab)=2$ since $ab$ appear in $abcbab$ twice in position $1$ and $5$.

## Modified precision score

We start from the one candidate translation $\hat{y}$ and one reference translation $y$. The modified $n$-gram is defined as

$$ p_n(\hat{y}, y) = \frac{\sum_{s\in G_n(\hat{y})}\min (C(s,\hat{y}), C(s,y))}{\sum_{s\in G_n(\hat{y})}C(s,\hat{y})} $$

The quantity measures how many $n$-grams of the reference translation $y$ appears in the candidate translation $\hat{y}$.
In case that $\hat{y}$ is too short, we take a minimum between $C(s,\hat{y})$ and $ C(s,y)$. Then we normalize to make $p_n(\hat{y}, y)$ comparable among multiple translation pairs.

Now, suppose we have a candidate translation corpus, $\hat{S}=\{\hat{y}^1,\dots,\hat{y}^M\}$, and for each candidate translation $\hat{y}^i$, we have a reference translation corpus (there are multiple translations can represent the same meaning) $S_i=\{y^{i,1},\dots,y^{i,N_i}\}$. We define $S=\{S_1,\dots,S_M\}$, then our modified $n$-gram precision is defined as

$$ p_n(\hat{S}, S) = \frac{\sum_{i=1}^M\sum_{s\in G_n(\hat{y})}\min (C(s,\hat{y}), \max_{y\in S_i}C(s,y))}{\sum_{i=1}^M\sum_{s\in G_n(\hat{y})}C(s,\hat{y})} $$

note that we have replaced $\min (C(s,\hat{y}), C(s,y))$ with $\min (C(s,\hat{y}), \max_{y\in S_i}C(s,y))$ since there are multiple reference translation, we use the most similar one. So this is to say: "There are multiple answer, go to choose the best one and compute the score."

## BP (Brevity Penalty)

Candidate translations longer than their references are already penalized by the modified $n$-gram precision measure.
Now, to penalize those translations shorter than the reference translations, we need add an penalty term. This is when brevity penalty comes out:

$$
\mathrm{BP}=\begin{cases}1&\text{ if }c > r\\
\exp(1-\frac{r}{c})&\text{ if }c \leq r
\end{cases}
$$

where

- $c$ is the number of words or tokens of the candidate corpus. That is,

$$ c = \sum_{i=1}^M\mathrm{length}(\hat{y}^i) $$

- $r$ is the number of words or tokens of the effective reference corpus length, where the effective reference is defined as the reference translation whose length is as close to the corresponding candidate translation as possible. That is

$$ r = \sum_{i=1}^M\mathrm{length}(y^{i,j}),\text{ where } y^{i,j}=\arg\min_{y\in S_i}|\mathrm{length}(\hat{y}^i)-\mathrm{length}(y)| $$

with this penalty term, we wish the model to output the translations with the same length as the reference translations.

## Weight

The weight measures the importance of different precision score, in the original paper, the uniform weights are adopted, that is

$$ w_i = \frac{1}{N}, \ \text{ for } i=1,\dots,N $$

## Final definition

The final definition of the BLEU is given by

$$ \mathrm{BLEU}_ {w}(\hat{S}, S) = \mathrm{BP} \cdot \exp\left(\sum_{n=1}^\infty w_n\log p_n(\hat{S}, S) \right) $$

usually, the upper-bound of the above summation can be reduced to $\max_{i=1,\dots,M}\mathrm{length}(\hat{y}^i)$.

# Analysis

Disadvantages of BLEU:

- BLEU compares overlap in tokens from the predictions and references, instead of comparing meaning. This can lead to discrepancies between BLEU scores and human ratings.
- BLEU scores are not comparable across different datasets, nor are they comparable across different languages.
- BLEU scores can vary greatly depending on which parameters are used to generate the scores, especially when different tokenization and normalization techniques are used.
- BLEU ignores synonym or similar expression, which causes refuses of reasonable translation.
- BLEU is affected by common words.

# Python Implementation

```python
import math
from typing import Set, List


def compute_n_gram_set(s: str, n: int) -> Set[str]:
    return set(s[i : i + n] for i in range(len(s) - n))


def modified_n_gram_precision(S_hat: List[str], S: List[List[str]], n: int) -> float:
    if len(S_hat) < n:
        return 0.0
    numerator: int = 0
    denominator: int = 0

    for index, y_hat in enumerate(S_hat):
        n_gram_set = compute_n_gram_set(y_hat, n)
        if not n_gram_set:
            continue
        # print(n_gram_set)
        for n_gram in n_gram_set:
            candidate_substr_count = y_hat.count(n_gram)
            best_ref_substr_count, best_ref_len = max(
                [(y.count(n_gram), len(y)) for y in S[index]], key=lambda x: x[0]
            )
            numerator += min(candidate_substr_count, best_ref_substr_count)
            denominator += candidate_substr_count

    return numerator / denominator


def brevity_penalty(S_hat: List[str], S: List[List[str]]) -> float:
    r: int = 0
    c: int = 0
    for index, y_hat in enumerate(S_hat):
        c += len(y_hat)
        best_match_ref = min(S[index], key=lambda x: abs(len(x) - len(y_hat)))
        r += len(best_match_ref)
    return 1.0 if c > r else math.exp(1 - r / c)


def compute_bleu_score(S_hat: List[str], S: List[List[str]]) -> float:
    assert len(S_hat) == len(S)

    # take N as a sufficiently large integer
    # N = max(len(y_hat) for y_hat in S_hat)
    N = sum(len(y_hat) for y_hat in S_hat)

    bp = brevity_penalty(S_hat, S)
    precisions = [modified_n_gram_precision(S_hat, S, n) for n in range(1, N + 1)]

    # bleu_score = bp * exp(p_n)
    return bp * math.exp(
        sum(math.log(precision) for precision in precisions if precision != 0)
    )
```

# Reference

- [Hugging Face space](https://huggingface.co/spaces/evaluate-metric/bleu)
- [Original Paper](https://aclanthology.org/P02-1040.pdf)
- [Wikipedia Documentation, Recommended](https://en.wikipedia.org/wiki/BLEU)

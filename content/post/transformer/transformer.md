---
title: Formal Algorithms for Transformer
description: An formal algorithm describing how transformer works.
date: 2024-05-02 13:13:12+0800

tags: 
    - transformer
categories:
    - LLM
math: true
---

This post is a notes on understanding how transformer works in an algorithm perspective.

# Introduction

Transformer is a neural architecture that is used for neural language processing. Transformer receives an embedding matrix, which represents a sentence as input, and outputs a matrix of the same size as the embedding matrix, then the output can be used for downstream tasks.

## Notation

1. We denote $V=[N_V]:=\{1,\dots,N_V\}$ as the *vocabulary* of tokens or words or characters.
2. We denote $\bm{x}=x[1...n]:=x[1]...x[n]\in V^n$ be a sequence of tokens, for example, a sentence or a paragraph.
3. Given a matrix $M\in\mathbb{R}^{m\times n}$, $M[i,:]\in\mathbb{R}^n$ is the $i$-th row of $M$, $M[:, j]\in\mathbb{R}^m$ is the $j$-th column of $M$.

# Tokenization

Tokenization determines how the text are represented. Given a piece of text, for example, `"I have an apple"`, we seek to  find a proper way to represent this sentence.

1. Character level tokenization. In this setting, $V$ is the English alphabet plus punctuation. This tends to yield very long sequences (depends on the character contained in the raw text).
2. Word level tokenization. In this setting, $V$ is the set of all English words plus punctuation. Word level tokenization is straightforward, but it tends to required a very large vocabulary and cannot handle new vocabulary at test time.
3. Subword tokenization. This is the most common way used in nowadays, $V$ is the set containing the commonly used word segments like "ing", "est". This can be computed via Byte-Pair Encoding (BPE) algorithm.
4. We suppose the length of the input text is $L$, if the length input text exceeds $L$, we chunk it.

After tokenization, each element in the vocabulary is assigned to a unique index $i\in\{1,\dots,N_V-3\}$, and a number of special tokens are added to the vocabulary. For example:

1. `mask_token`$=N_V-2$, used in masked language modeling
2. `bos_token`$=N_V-1$ and `eos_token`$=N_V$, these two tokens are used to represent the beginning and the end of the sequence.

Finally, a piece of raw text is represented as a sequence of indices, often called *token ID*s corresponding to its subwords, preceded by `bos_token` and followed by `eos_token`.

# Embedding

The embedding layer is used to represent each token as a vector that contains richer semantic information. The embedding contains two parts:

1. token embedding, where each token is embedded into a vector space
2. positional embedding, where embeds the position information of the tokens.

## Token embedding

Given a sequence of token ID, we now need to represent each token as a vector in $\mathbb{R}^d$.

The simplest way is to use *one-hot embedding*, where each token $i$ is represented a vector $[0,\dots,1,\dots,0]\in\mathbb{R}^{N_V}$ whose elements are all $0$ excepts that $i$-th position is equal to $1$. However, the problem is that the vocabulary size $N_V$ is two large.

To solve this problem, we can train a learnable embedding model, of which parameter is a matrix $W_e\in\mathbb{R}^{d\times N_V}$,  its $i$-th row corresponds to vector representation of the token $i$:

$$ \bm{e} = W_{e}[:, i]\in\mathbb{R}^d $$

## Position embedding

There is a problem in token embedding, that is, it doesn't contain consider the order of tokens. In latter, we show that the self-attention mechanism is equivariant to a permutation matrix $X\Pi$ of data $X$, where $\Pi$ is a permutation matrix, that is,

$$ \mathrm{Sa}(X\Pi) = \mathrm{Sa}(X)\Pi $$

the above equation indicates that the self-attention layer learn no position information at all!

To solve this problem, we add a positional embedding to token embedding. There are two kinds of embeddings:

1. Absolute positional embeddings. In this setting, a matrix $W_P\in\mathbb{R}^{d\times N}$ is learned or design to indicate the position of tokens. Mathematically, we have

$$ \bm{e}_p = W_p[:, \mathrm{index}(i)]\in\mathbb{R}^{d} $$

where $\mathrm{index}(i)$ is the index of token $i$ in the input sequence.
2. Relative positional embeddings. We leave this in latter notes. Compared to absolute positional embeddings, relative positional embeddings uses offset information, which performs well when the input sequence is too long.

The final embedding of a token $i$ is given by

$$ \bm{e} = W_e[:, i] + W_p[:, \mathrm{index}(i)]\in\mathbb{R}^{d} $$

# Attention

The idea of attention mechanism is: Given a sequence of token, to predict the current token, which token should I pay attention to? For example, `I opened the door with my ___`, we may answer `key`, `password` or `fingerprint` etc. This is because we notice that we `opened the door`, so to predict the next token, we should make use of the information. What attention mechanism does is to quantify this process and make them parallel and learnable.

## Single query attention

We first consider a simple example. Given the embedding of the current token $\bm{e}\in\mathbb{R}^d$ and the list of context tokens $[\bm{e}_1,\dots,\bm{e}_N]\in\mathbb{R}^{d\times N}$, the attention is given as follows:

1. compute query vector: $\bm{q}=W_q\bm{e}+b_q\in\mathbb{R}^{d}$
2. compute key vectors: for $i=1,\dots,L$, $\bm{k}_i=W_k\bm{e}_i+b_k\in\mathbb{R}^{d}$
3. compute value vectors: for $i=1,\dots,L$, $\bm{v}_i=W_v\bm{e}_i+b_v\in\mathbb{R}^{d}$
4. compute attention weights: let $\bm{s}=[\bm{q}^T\bm{k}_1,\dots,\bm{q}^T\bm{k}_L]\in\mathbb{R}^{N}$, then:

$$ \bm\alpha = \mathrm{softmax}\left(\frac{\bm{s}}{\sqrt{d}}\right)\in\mathbb{R}^{N}$$
5. compute vector representation of the token and context combined:

$$ \bm{v}'= \sum_{i=1}^N\alpha_i\bm{v}_i\in\mathbb{R}^{d} $$

where  $W_q,W_k,W_v\in\mathbb{R}^{d\times d}$, $b_q,b_k,b_v\in\mathbb{R}$.

## General attention

To extend the single query attention to general form, we consider the embedding matrix $X\in\mathbb{R}^{D\times N}$, the context matrix $Z\in\mathbb{R}^{d\times C}$ and a mask matrix $M\in\mathbb{R}^{D\times D}$, then the attention is computed as follows:

1. compute query matrix:

$$ Q=W_qX+\bm{b}_q\in\mathbb{R}^{D\times N}$$
2. compute key matrix:

$$ K=W_kZ+\bm{b}_k\in\mathbb{R}^{D\times C}$$
3. compute value vectors:

$$ V=W_vZ+\bm{b}_v\in\mathbb{R}^{D\times C}$$
4. compute attention weights:

$$\mathrm{Sa}(X) = \mathrm{softmax}\left(M\odot \frac{K^TQ}{\sqrt{D}}\right) \in\mathbb{R}^{C\times N} $$
where $\odot$ is the element-wise product.
5. output the updated representations of tokens in $X$, folding the information from tokens in $Z$

$$ \tilde{V} = V\odot \mathrm{Sa}(X)\in\mathbb{R}^{D\times N} $$

There are two kinds of mask matrices depending on which attention we are using:

1. Bidirectional attention, in this case, $M=\bm{1}\bm{1}^T\in\mathbb{R}^{C\times N}$.
2. Undirectional attention, in this case, $M[i,j] = \bm{1}_{i\leq j}$, where $\bm{1}_{i\leq j}$ is the *indicator function*.

## Multi-head Attention

The previous describes the operation of a *single head*. In practice, transformers run multiple attention heads in parallel and combine their outputs, this is called *multi-head attention*. The idea behind multi-head attention can be summarized as follows:

1. In high dimensional space, two vectors are usually far from each other, with multiple attention heads, we can reduce the dimension of the representation.
2. With multiple attention heads, each heads may focus on specific semantics of the representation. For example, one head focuses on positiveness, and one another head focuses on noun/verb semantics.

For simplicity, we denote the single-head attention as $\mathrm{attention}(X, Z\mid M)$, suppose we have $H$ heads, then we compute $\tilde{V}_i$ for each heads:

$$ \tilde{V}_i = \mathrm{attention}(X, Z\mid M)\in\mathbb{R}^{D\times N},i=1,\dots,H $$
then attention representation are concatenated together:

$$ V = [\tilde{V}_1^T, \dots, \tilde{V}_H^T]^T\in\mathbb{R}^{HD\times N} $$

combined via an output matrix $W_o\in\mathbb{R}^{D\times HD}$:

$$ \tilde{V} = W_oV+\bm{b}_o\in\mathbb{R}^{D\times N}  $$

We denote the multi head attention as $\mathrm{MhSa}(X, Z\mid M)$.

## Transformer layer

After computing the multi head attention, we can now construct a transformer layer, which can also be stacked as convolution neural networks. A transformer layer can be constructed by the following operations:

1. Multi head attention (residual), $X\gets X + \mathrm{MhSa}(X, Z\mid M)$.
2. Layer norm, $X \gets \mathrm{LayerNorm}(X)$
3. Multi layer perception $\bm{x}_i\gets \bm{x}_i + \mathrm{mlp}(\bm{x}_i), i=1,\dots,N$
4. Layer norm, $X \gets \mathrm{LayerNorm}(X)$

where $\mathrm{LayerNorm}$ is the layer norm operation. $\mathrm{mlp}$ is a multi layer perception, usually it consists of one hidden layer of size $4D$, that is, then umber of neurons in three layers are $D, 4D, D$.

Usually, a large language model consists of multiple transformer layers.

# Unembedding

The unembedding learns to convert a vector representation of a token and its context $\bm{e}$ into a distribution over the vocabulary elements.

$$ \bm{p} = \mathrm{softmax}(W_u\bm{e})\in \Delta(V)\subseteq \mathbb{R}^d $$
where $\Delta(V)$ is a simplex over the set $V$.

# Reference

- [Understanding Deep Learning Chapter 12](https://udlbook.github.io/udlbook/)
- [Formal Algorithms for Transformers](http://arxiv.org/abs/2207.09238)

---
title: ROUGE (Recall-Oriented Understudy)
date: 2024-05-09 17:35:20+0800
description: The metric that evaluates similarity between summaries.
tags: 
    - Metric
categories:
    - LLM 
    - NLP
math: true
---

ROUGE stands for Recall-Oriented Understudy for Gisting Evaluation. It includes several automatic evaluation methods that measure the similarity between summaries.

# Preliminaries

# ROUGE-N: N-gram co-occurrence statistics

## Definitions

Given any string $y=y_1y_2\cdots y_K$, where $y_i,i\in{1,\dots,K}$ are characters and an integer $n\geq1$, we define the **set of $n$-gram** to be

$$ G_n(y) = \{ y_1\cdots y_n, y_2\cdots y_{n+1}, \dots, y_{K-n+1}\cdots y_K\} $$

NOTE that this is a set with unique elements, for example, $G_2(abab)=\{ab, ba\}$.

Given any two strings $s$ and $y$, we define **substring count** $C(s,y)$ to tbe the number of appearances of $s$ as a substring of $y$. For example, $C(ab, abcbab)=2$ since $ab$ appear in $abcbab$ twice in position $1$ and $5$.

## Base Version

ROUGE-N is an n-gram recall between a candidate summary $\hat{y}$ and a set of reference summaries $S=\{y_1,\dots,y_n\}$. ROUGE-N is defined as follows:

$$ \text{ROUGE-N}(\hat{y}, S) = \frac{\sum_{i=1}^n\sum_{s\in G_N(y_i)}C(s,\hat{y})}{\sum_{i=1}^n\sum_{s\in G_N(y_i)}C(s, y_i)} $$

Features of ROUGE-N:

1. the denominator increases as we add more references, since there might exists multiple good summaries.
2. A candidate summary that contains words shared by more references is favored by the ROUGE-N measure.

## Multiple references

When there are multiple references for a candidate summary, it is suggests to use the following formula:

$$ \text{ROUGE-N}(\hat{y}, S) = \arg\max_{i=1,\dots,n}\text{ROUGE-N}(\hat{y}, \{y_i\})  $$

the above formula is favored for the following reasons:

1. There is no single "best" reference summary, the multiple reference formula allows ROUGE-N to take into account of all the possible reference summaries and provide a more accurate measure of the quality of the generated summary.
2. The multiple reference formula is more robust. If a reference summary contains a typo or a grammatical error, this can affect the ROUGE-N score.
3. The multiple reference formula can provide a more comprehensive evaluation of the generated summary, since it can allow ROUGE-N to evaluate the generated summary against a wider range pf possible reference summaries.

# ROUGE-L

A sequence $Z=[z_1,\dots,z_m]$ is a subsequence of another sequence $X=[x_1,\dots,x_n]$ if there exists a strict increasing sequence $[i_1,\dots,i_k]$ of indices of $X$ such that for all $j=1,\dots,k$, we have $x_{i_j}=z_j$.

Given two sequences $X$ and $Y$, the longest common subsequences (LCS) of $X$ and $Y$ is a common subsequences with maximum length.

```python
def longest_common_subsequence(x: str, y: str) -> int:
    """compute the length of LCS of x and y

    Args:
        x (str): a string of length m
        y (str): a string of length n
    
    Return:
        int: the length of LCS of x and y
    """
    m, n = len(x), len(y)
    dp = [[0 for _ in range(m + 1)] for _ in range(n + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i] == y[j]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    return dp[m][n]

```

## Sentence-level LCS

The intuition of sentence-level LCS is that the longer the LCS of two summary sentences is, the more similar the two summaries are.

Given two summaries $X$ of length $m$ and $Y$ of length $n$, assuming $X$ is a reference summary sentence and $Y$ is a candidate summary sentence, the LCS-based recall, precision and F-measure are defined as follows:

$$ R_{LCS} = \frac{LCS(X, Y)}{m}, P_{LCS} = \frac{LCS(X, Y)}{n}, R_{LCS} = \frac{(1+\beta^2)R_{LCS}P_{LCS}}{R_{LCS}+\beta^2P_{LCS}} $$

the above formula is called ROUGE-L. $\beta$ is a hyperparameter.

Features of ROUGE-L are listed as follows:

1. It doesn't require consecutive matches but in-sequence matches, which is more reasonable than n-grams.
2. It automatically includes longest in-sequence common n-grams, there fore no predefined n-gram length is necessary.
3. It's value is less tan or equal to the minimum of unigram F-measure of $X$ and $Y$.
4. The disadvantage of ROUGE-L is that it only counts the main in-sequences words, therefore, other alternative LCSes and shorter sequences are not reflected in the final score.

## Summary-level LCS

We can apply sentence-level LCS-based F-measure score to summary level. Given a reference summary of $u$ sentences $\{r_1,\dots,r_u\}$ containing a total of $m$ words and a candidate summary of $v$ sentences $\{c_1,\dots,c_v\}$ containing a total of $n$ words, the summary-level LCS-based recall, precision and F-measure are defined as follows:

$$ R_{LCS} = \frac{\sum_{i=1}^u\max_{j}LCS(r_i, c_j)}{m}, P_{LCS} = \frac{\sum_{i=1}^v \max_{j}LCS(r_i, c_j)}{n}, R_{LCS} = \frac{(1+\beta^2)R_{LCS}P_{LCS}}{R_{LCS}+\beta^2P_{LCS}} $$

# ROUGE-W

The basic LCS has a problem that it doesn't differentiate LCSes of different spatial relations within their embedding sequences.

To improve the basic LSC method, we can simply remember the length of consecutive matches encountered so fat to a regular two dimensional dynamic program table computing LCS. we call this *weighted LCS (WLCS)* and use $k$ to indicate the length of the current consecutive matches ending at words $x_i$ and $y_j$.

```python
def weighted_longest_common_subsequence(x: str, y: str, f: Callable):
    """use dynamic programming to compute WLCS with a weighted function f

    Args:
        x (str): a candidate summary containing m words
        y (str): a reference summary containing n words
        f (Callable): a function satisfies f(x+y) > f(x) + f(y)
    
    Return:
        the WLCS score

    Reference:
        https://aclanthology.org/W04-1013
    """
    x, y = x.split(), y.split()
    m, n = len(x), len(y)
    c = [[0 for _ in range(m + 1)] for _ in range(n + 1)]
    w = [[0 for _ in range(m + 1)] for _ in range(n + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i] == y[j]:
                # the length of consecutive matches at (i - 1, j - 1)
                k = w[i - 1][j - 1]
                c[i][j] = c[i - 1][j - 1] + f(k + 1) - f(k)
                # remember the length of consecutive matches at (i - 1, j - 1)
                w[i][j] = k + 1
            else:
                if c[i - 1][j] > c[i][j - 1]:
                    c[i][j] = c[i - 1][j]
                    w[i][j] = 0     # no match at (i, j)
                elif c[i][j] = c[i][j - 1]:
                    w[i][j] = 0     # no match at (i, j)

    return c[m][n]
```

where `c[i][j]` stores the WLCS score ending at word `x[i]` of `x` and `y[i]` of `y`. `w` stores the length of consecutive matches at `c[i][j]`. `f` is a function of consecutive matches at `c[i][j]`.

Recall, precision, F-score based on WLCS can be computed as follows:

$$ R_{WLCS} = f^{-1}\left(\frac{WLCS(X, Y)}{f(m)}\right), P_{WLCS} = f^{-1}\left(\frac{WLCS(X, Y)}{f(n)}\right), R_{LCS} = \frac{(1+\beta^2)R_{WLCS}P_{WLCS}}{R_{WLCS}+\beta^2P_{WLCS}} $$

where $f$ is the inverse function of $f$. We call the WLCS-based F-measure as ROUGE-W. Usually, a function $f$ that has a close form inverse is preferred.

# ROUGE-S

Skip-bigram is any pair of words in their sentence order, allowing for arbitrary gaps. Skip-bigram co-occurrence statistics measure the overlap of skip-bigrams between a candidate translation and a set of reference translations.
A sentence with $n$ words will have $\binom{n}{2}=n(n-1)/2$ skip-bigrams.

Recall, precision, F-score based on skip-bigram can be computed as follows:

$$ R_{\mathrm{SKIP2}} = \frac{\mathrm{SKIP2}(X, Y)}{m}, P_{\mathrm{SKIP2}} = \frac{\mathrm{SKIP2}(X, Y)}{n}, R_{\mathrm{SKIP2}} = \frac{(1+\beta^2)R_{\mathrm{SKIP2}}P_{\mathrm{SKIP2}}}{R_{\mathrm{SKIP2}}+\beta^2P_{\mathrm{SKIP2}}} $$

where $\mathrm{SKIP2}(X, Y)$ is the number of skip-bigram matches between $X$ and $Y$. The F-score is called ROUGE-S.

# ROUGE-SU

One problem of ROUGE-S is that it doesn't given any credit to a candidate sentence if the sentence doesn't have any word pair co-occurring with its references.

To fix this problem, we extend ROUGE-S with the addition of unigram as counting unit. The extended version is called ROUGE-SU. We can also obtain ROUGE-SU from ROUGE-S by adding a begin-of-sentence marker at the beginning of candidate and reference sentences.

# Reference

- [ROUGE: A Package for Automatic Evaluation of Summaries](https://aclanthology.org/W04-1013.pdf)
- [ROUGE Eval](https://github.com/Yale-LILY/SummerTime/blob/e49058d928b4bd5b1017b7d774bea984bbdf5006/summertime/model/third_party/HMNet/Evaluation/OldROUGEEval.py)
- [google-research rouge](https://github.com/google-research/google-research/tree/master/rouge)

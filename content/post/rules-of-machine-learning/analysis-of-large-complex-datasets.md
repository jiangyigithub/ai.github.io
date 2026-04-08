---
title: Practical advice for analysis of large, complex data sets
description: Advice on how to analyze complex and large data sets
date: 2024-04-17 22:40:11+0800

categories:
    - machine learning
---

# Introduction
These advises are given by Patrick Riley in 2016, though it has been years util now, I think some of them are still useful.

The advice is organized into three general areas:
- Technical: Ideas and techniques for how to manipulate and examine your data.
- Process: Recommendation on how you approach your data, what questions to ask, and what things to check.
- Social: How to work with others and communicate about your data and insights.

# Technical
## Look at your distribution
Besides the typically used summary metrics, we should looking at a much richer representation of the distribution, such as histograms, CDFs, Q-Q plots, etc. This allows us to see some interesting features.

## Consider the outliers
We should look at the outliers in our data. It's fine to exclude them from our data or to lump them together into an unusual category, but we should make sure we know why.

## Report noise/ confidence
Every estimator that you produce should have a notion of your confidence in this estimate attached to it.

## Look at examples
Anytime you are producing new analysis code, you need to look at examples of the underlying data and how your code is interpreting those examples.

## Slice your data
slicing help us obtain underlying features of the data subgroups easier. However, when we use slicing, we need to care about the mix shift.

## Consider practical significance
Don't be blind by statistics, watch out those may have an impact on deploying or ethical problems.

## Check for consistency over time
One particular slicing you should almost always employ is to slice by units of time. 
This is because many disturbances to underlying data happen as our systems evolve over time. 

# Process

## Separate Validation, description, and evaluation
- Description should be things that everyone can agree on from the data.
- Evaluation is likely to have much more debate because you imbuing meaning and value to the data.

## Confirm expt/data collection setup
Before looking at any data, make sure you understand the experiment and data collection setup

## Check vital signs
Before actually answering the question you are interested in you need to check for a lot of other things that may not be related to what you are interested in but may be useful in later analysis or indicate problems in the data

## Standard first, custom second
When we use metric, we should always look at standard metrics first, even if we expect them to change.

## Measure twice, or more
If you are trying to capture a new phenomenon, try to measure the same underlying thing in multiple ways.
Then, check to see if these multiple measurements are consistent

## Check for reproducibility
Both slicing and consistency over time are particular examples of checking for reproducibility. 
If a phenomenon is important and meaningful, you should see it across different user populations and time. 

## Check for consistency with past measurements
You should compare your metrics to metrics reported in the past, even if these measurements are on different user populations.

New metrics should be applied to old data/features first


## Make hypothesis and look for evidence
Typically, exploratory data analysis for a complex problem is iterative. You will discover anomalies, trends, or other features of the data. Naturally, you will make hypotheses to explain this data. It’s essential that you don’t just make a hypothesis and proclaim it to be true. Look for evidence (inside or outside the data) to confirm/deny this theory.

## Exploratory analysis benefits from end to end iteration
When doing exploratory analysis, you should strive to get as many iterations of the whole analysis as possible.

# Social

## Data analysis starts with questions, not data or a technique
Ask question first and use tools to answer the questions.

## Acknowledge and count your filtering
- Acknowledge and clearly specify what filtering you are doing
- Count how much is being filtered at each of your steps
the best way to do the latter is to actually compute all your metrics even for the population you are excluding

## Ratios should have clear numerator and denominators
When you communicate results containing ratios, you must be clear about the numerator and denominator.

## Educate your consumers
You are responsible for providing the context and a full picture of the data and not just the number a consumer asked for.


## Be both skeptic and champion
As you work with data, you must be both the champion of the insights you are gaining as well as a skeptic. 

## Share with peers first, external consumers second
A skilled peer reviewer can provide qualitatively different feedback and sanity-checking than the consumers of your data can, especially since consumers generally have an outcome they want to get


# Reference
- [Practical advice for analysis of large, complex data sets](https://www.unofficialgoogledatascience.com/2016/10/practical-advice-for-analysis-of-large.html)
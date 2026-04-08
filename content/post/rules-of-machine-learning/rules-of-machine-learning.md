---
title: Rules of Machine Learning
description: Advice for machine learning
date: 2024-04-13 19:57:47+0800

categories:
    - Machine Learning 
---

Best practice for machine learning.

# Introduction
Google posts a guide on how to uses machine learning in practice.
It represents a style for machine learning, similar to Google C++ Style Guide.

In overview, to make great products:
> Do machine learning like the great engineer your are, not like the great machine learning expert you are.

Most algorithms we are facing are engineering problems instead of machine learning algorithms. A basic approach Google recommends is:
1. Make sure your pipeline is solid end to end
2. Start with a reasonable objective
3. Add a common-sense features in a simple way
4. Make sure that your pipeline stays solid.

Google separates rules with respect to different stages.

# Before machine learning
These rules help us understand whether the time is right for building a machine learning system.

> Rule #1: Do not be afraid of lunch a product without machine learning.

This rule tells us that is not absolutely necessary, if rule-based methods work well, there is no need to develop a machine learning algorithm.

> Rule #2: First, design and implement metrics.

This rule tells us that tracking as much as possible before we formalize what our machine learning system will do. This step helps us construct the goal of our system, that is, a metric.

> Rule #3: Choose machine learning over a complex heuristic.

Considering using machine learning algorithms only if the heuristic algorithm doesn't work well, since a complex heuristic is not maintainable. Meanwhile, machine-learned models are easier to update and maintain.

# ML phase 1: Your First Pipeline
When creating our first pipeline, we should focus on our system infrastructure

> Rule #4: Keep the first model simple and get the infrastructure right.

Remember infrastructure issues are many more than model problems when we creating the first machine learning model. We have to determine:

1. How to obtain data for our model
2. How to evaluate the performance of our model
3. How to integrate our model into our application

Moreover, we should choose simple features to ensure that:
- The features reach our learning algorithm correctly
- The model learns reasonably weights
- The features reach our model in the sever correctly

Once we have a system that does above things reliably, we have done most of the work.

> Rule #5: Test the infrastructure independently from the machine learning.

This rule tells us split the infrastructure and the machine learning model and test them separately to avoid dependency issue occurs. In this way, our model can be developed without worrying about environment. Specifically:
1. Test getting data into algorithm. Check the data that are feed into the models and do some statistics before using the data
2. Testing getting models out of the training algorithm. Make sure our algorithms work in the same way in our serving environment as the training environment.

> Rule #6: Be careful about dropped data.

Do not drop data without thoroughly test, this may data loss.

> Rule #7: Turn heuristics into features, or handle them externally.

Before using machine learning models, if we have tried some heuristic algorithms, then these algorithms may help us to improve the overall performance. Some ways we can use an existing heuristic algorithm:
1. Preprocessing using the heuristic. If the feature is incredibly awesome, then do not try to relearn the feature. Just use the heuristic way to pre-process the data.
2. Create a feature. We can use the heuristic way to create a new feature to help improve the machine learning performance.
3. Mine the raw inputs of the heuristic. We can use the inputs of heuristic as features to learn the heuristic implicitly. 
4. Modify the label. 

## Monitoring
In general, such as making alerts and having a dashboard page.

> Rule #8: Know the freshness requirements of our system.

It is important for us to know the freshness of our model, for example, how much does performance degrade if we have a model that is a day old. The freshness helps us monitor and improve the performance.

> Rule #9: Detect problems before exporting models.

This rule tells us that evaluating the performance of the model before serving. The evaluation includes the testing on hold-out data, check AUC metric.

> Rule #10: Watch for silent failures.

Since the continuos change of data, silent failures may occur, so keep tracking statistics of the data as well as manually inspect the data on occasion help us reduce these kind of issues.

> Rule #11: Give feature owners and documentation.

Knowing who created the feature helps us gain information about data. A detailed documentation helps user understand how it works.

## Your first objective

> Rule #12: Don't overthink which objective you choose to optimize.

There are many metrics to optimize according to Rule #2. However, it turns out in early stage, some metrics are optimized even though we not directly optimizing them.

> Rule #13: Choose simple, observable and attributable metric for your first objective.

This rule tells us the strategy of choosing metric in the beginning. In principal, The ML objective should be something that is easy to measure and is a proxy for the "true" objective. In fact however, there is no such "true" objective, so we should keep the objective as simple as possible, it's better if the objective is observable. Then, we can modify the objective based on the performance of the model.

> Rule #14: Starts with an interpretable model makes debugging easier.

> Rule #15: Separate spam filtering and quality ranking in a policy layer.

Sometimes, spam filtering confuses quality ranking, when we do quality ranking, we should clean the data.


# ML phase 2: Feature engineering
After we have a working end to end system with unit and system tests instrumented, Phase II begins.
In this phase, we should make use of features.

> Rule #16: Plan to lunch and iterate.

There are three reasons to lunch new models:
1. You are coming up with new features
2. You are tuning regularization and combining old features in new ways
3. You are tuning the objectives

When lunch a new model, we should think about:
1. How easy is it to add or remove or recombine features
2. How easy is it to create a fresh copy of the pipeline and verify its correctness.
3. Is it possible to have two or three copies running in parallel.

> Rule #17: Start with directly observed and reported features as opposed to learned features.

a learned feature is a feature generated by an external system or by the learner itself.
If the learned feature comes from an external system, then bias or being out-of-date may affect the model.  If the learned feature comes from the learner itself, then it is hard to tell the impact of the feature.

> Rule #18: Explore with features of content that generalize across contexts.

> Rule #19: Use very specific features when you can.

It is simpler to learn millions of simple features than a few complex features.

> Rule #20: Combine and modify existing features to create new features in human-understanding ways.

Combine features may causing overfitting problems

> Rule #21: The number of feature weights we can learn in a linear model is roughly proportional to the number of data you have.

> Rule #22: Clean up features you are no longer using.

If you find that you are not using a feature, and that combining it with other features is not working, then drop it out of your infrastructure. 

## Human analysis of the system
This subsection teaches us how to look at an existing model and improve it.

> Rule #23: You are not a typical end user.

Check carefully before we deploying the model.

> Rule #24: Measure the delta between models

Make sure the system is stable when making small changes. Make sure that a model when compared with itself has a low (ideally zero) symmetric difference.

> Rule #25: When choosing models, utilitarian performance trumps predictive power.

> Rule #26: Look for patterns in the measured errors, and create new features.

Once you have examples that the model got wrong, look for trends that are outside your current feature set.

> Rule #27: Try to quantify observed undesired behavior

If your issues are measurable, then you can start using them as features, objectives, or metrics. The general rule is "**measure first, optimize second**".

> Rule #28: Be aware that identical short-term behavior does not imply identical long-term behavior.

## Training-Serving skew
Training-serving skew is a difference between performance during training and performance during serving. 
This skew can be caused by:
1. A discrepancy between how you handle data in the training and serving pipelines
2. A change in the data between when you train and when you serve
3. A feedback loop between your model and your algorithm.

The best solution is to explicitly monitor it so that system and data changes don't introduce skew unnoticed.

> Rule #29: The best way to make sure that you train like you serve is to save the set of features used at the serving time, and then pipe those features to a log to use them at training time.

This can help verify the consistency between the training and serving.

> Rule #30: Importance-weight sampled data, don't arbitrarily drop it.

Importance weighting means that if you decide that you are going to sample example X with a 30% probability, then give it a weight of 10/3. With importance weighting, all of the calibration properties discussed in Rule #14 still hold.

> Rule #31: Beware that if your join data from a table at training and serving time, the data in the table may change.

> Rule #32: Re-use code between your training pipeline and your serving pipeline whenever possible.

> Rule #33: If you produce a model based on the data until January 5th, test the model on the data from January 6th and after.

In general, measure performance of a model on the data gathered after the data you trained the model on, as this better reflects what your system will do in production

> Rule #37: Measure Training-Serving Skew.

We can divide causes of Training-Serving Skew into several parts:
1. The difference between the performance on the training data and the holdout data. In general, this will always exist, and it is not always bad.
2. The difference between the performance on the holdout data and the "next­day" data. Again, this will always exist
3. The difference between the performance on the "next-day" data and the live data. 

# ML phase 3: Slowed growth, Optimization refinement, and complex models

> Rule #38: Don’t waste time on new features if unaligned objectives have become the issue.

> Rule #39: Launch decisions are a proxy for long-term product goals.

The only easy launch decisions are when all metrics get better (or at least do not get worse).

Individuals, on the other hand, tend to favor one objective that they can directly optimize.

> Rule #40: Keep ensembles simple.

To keep things simple, each model should either be an ensemble only taking the input of other models, or a base model taking many features, but not both.

> Rule #41: When performance plateaus, look for qualitatively new sources of information to add rather than refining existing signals.

As in any engineering project, you have to weigh the benefit of adding new features against the cost of increased complexity.

> Rule #42: Don’t expect diversity, personalization, or relevance to be as correlated with popularity as you think they are.

> Rule #43: Your friends tend to be the same across different products. Your interests tend not to be.

# Conclusion
In early stage, make sure that the infrastructure is well constructed, the used model can be simple.

In main stage, focusing on the utilitarian performance and the gap between training data and test data.

When utilizing features, use simple, observable features.

When deploying models, watch out training-serving skew.

# Reference
- [Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml)
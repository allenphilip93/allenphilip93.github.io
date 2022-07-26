---
title: Designing Machine Learning Systems
date: 2022-07-26 15:05:00 +0530
categories: [ML]
tags: [Notes, Learning]
math: true
pin: false
---

# Reference

[Chapter 2 - Designing Machine Learning Systems by Chip Huyen](https://learning.oreilly.com/library/view/designing-machine-learning/9781098107956/ch02.html)

# Introduction

ML systems design takes a system approach to MLOps, which means that we’ll consider an ML system holistically to ensure that all the components—the business requirements, the data stack, infrastructure, deployment, monitoring, etc.—and their stakeholders can work together to satisfy the specified objectives and requirements.

If this system is built for a business, it must be driven by business objectives, which will need to be translated into ML objectives to guide the development of ML models.

# Business & ML Objectives

When working on an ML project, data scientists tend to care about the ML objectives: the metrics they can measure about the performance of their ML models such as accuracy, F1 score, inference latency, etc. They get excited about improving their model’s accuracy from 94% to 94.2% and might spend a ton of resources—data, compute, and engineering time—to achieve that.

But the truth is: most companies don’t care about the fancy ML metrics. They don’t care about increasing a model’s accuracy from 94% to 94.2% unless it moves some business metrics.

While most companies want to convince you otherwise, the sole purpose of businesses, according to the Nobel-winning economist Milton Friedman, is to maximize profits for shareholders.
> The ultimate goal of any project within a business is, therefore, to increase profits, either directly or indirectly: directly such as increasing sales (conversion rates) and cutting costs; indirectly such as higher customer satisfaction and increasing time spent on a website.

For an ML project to succeed within a business organization, it’s crucial to tie the performance of an ML system to the overall business performance.

One of the reasons why predicting ad click-through rates and fraud detection are among the most popular use cases for ML today is that it’s easy to map ML models’ performance to business metrics: every increase in click-through rate results in actual ad revenue, and every fraudulent transaction stopped results in actual money saved.

At times even rigorous experiments might not be sufficient to understand the relationship between an ML model’s outputs and business metrics. Imagine you work for a cybersecurity company that detects and stops security threats, and ML is just a component in their complex process. An ML model is used to detect anomalies in the traffic pattern. These anomalies then go through a logic set (e.g., a series of if-else statements) that categorizes whether they constitute potential threats. These potential threats are then reviewed by security experts to determine whether they are actual threats. Actual threats will then go through another, different process aimed at stopping them. When this process fails to stop a threat, it might be impossible to figure out whether the ML component has anything to do with it.

## Realistic Expectations

When evaluating ML solutions through the business lens, it’s important to be realistic about the expected returns. Due to all the hype surrounding ML, generated both by the media and by practitioners with a vested interest in ML adoption, some companies might have the notion that ML can magically transform their businesses overnight.

Magically: possible. Overnight: no.

There are many companies that have seen payoffs from ML. For example, ML has helped Google search better, sell more ads at higher prices, improve translation quality, and build better Android applications. But this gain hardly happened overnight. Google has been investing in ML for decades.

Returns on investment in ML depend a lot on the maturity stage of adoption. The longer you’ve adopted ML, the more efficient your pipeline will run, the faster your development cycle will be, the less engineering time you’ll need, and the lower your cloud bills will be, which all lead to higher returns.

![image](https://user-images.githubusercontent.com/20537002/180976940-ab0a6ce5-3daa-4eeb-8c25-88ca314f4185.png)

# Requirements for ML Systems

## Reliability

The system should continue to perform the correct function at the desired level of performance even in the face of adversity (hardware or software faults, and even human error).

Unlike traditional software systems, ML systems differ don't have to crash or have a system error in order to be labelled not reliable. What if the model's predictions have degraded? So it's important to note that ML systems can fail quietly!

## Scalability

There are multiple ways a ML system can grow in complexity:
- It can grow in complexity of the model
- It's traffic volume can grow
- It's model count can grow

An indispensable feature in many cloud services is autoscaling: automatically scaling up and down the number of machines depending on usage. However, handling growth isn’t just resource scaling, but also **artifact management**. With one model, you can, perhaps, manually monitor this model’s performance and manually update the model with new data. Since there’s only one model, you can just have a file that helps you reproduce this model whenever needed. However, with one hundred models, both the monitoring and retraining aspect will need to be automated. You’ll need a way to manage the code generation so that you can adequately reproduce a model when you need to.

## Maintainability

Code should be documented. Code, data, and artifacts should be versioned. Models should be sufficiently reproducible so that even when the original authors are not around, other contributors can have sufficient contexts to build on their work. When a problem occurs, different contributors should be able to work together to identify the problem and implement a solution without finger-pointing.

## Adaptability

To adapt to shifting data distributions and business requirements, the system should have some capacity for both discovering aspects for performance improvement and allowing updates without service interruption.

> Because ML systems are part code, part data, and data can change quickly, ML systems need to be able to evolve quickly. This is tightly linked to maintainability.

# Iterative Process

Developing an ML system is an iterative and, in most cases, never-ending process.10 Once a system is put into production, it’ll need to be continually monitored and updated.

For example, here is one workflow that you might encounter when building an ML model to predict whether an ad should be shown when users enter a search query:
- Choose a metric to optimize. For example, you might want to optimize for impressions—the number of times an ad is shown.
- Collect data and obtain labels.
- Engineer features.
- Train models.
- During error analysis, you realize that errors are caused by the wrong labels, so you relabel the data.
- Train the model again.
- During error analysis, you realize that your model always predicts that an ad shouldn’t be shown, and the reason is because 99.99% of the data you have have NEGATIVE labels (ads that shouldn’t be shown). So you have to collect more data of ads that should be shown.
- Train the model again.
- The model performs well on your existing test data, which is by now two months old. However, it performs poorly on the data from yesterday. Your model is now stale, so you need to update it on more recent data.
- Train the model again.
- Deploy the model.
- The model seems to be performing well, but then the businesspeople come knocking on your door asking why the revenue is decreasing. It turns out the ads are being shown, but few people click on them. So you want to change your model to optimize for ad click-through rate instead.
- Go to step 1.

The figure below shows that the process of developing a ML system looks more like a cycle with back and forth steps than a simple straight set of steps.
![image](https://user-images.githubusercontent.com/20537002/180980458-9c6b24b0-16e7-4a94-8def-3fa9e72d6872.png)

# Framing ML Problems

Before setting out to build our ML system, we need a business objective to work towards. Once we have that we need to see if ML can be used to address the problem and then tie our business objectives to our ML metrics. 

Framing the ML problems ties closes to the following:
- ML model's output
- ML model's objective function

## Model's output

We need to ask the following questions before deciding on what is the ML problem that we want to solve:
- Is this a problem of regression or classification?
- Do we need binary or multiclass classification?
- Do we need multiclass or multilabel classification?
- Can we reframe the problem between one of the above types to better solve the problem?

## Model's objective function

To learn, an ML model needs an objective function to guide the learning process. An objective function is also called a loss function, because the objective of the learning process is usually to minimize (or optimize) the loss caused by wrong predictions. For supervised ML, this loss can be computed by comparing the model’s outputs with the ground truth labels using a measurement like root mean squared error (RMSE) or cross entropy.

> Framing ML problems can be tricky when you want to minimize multiple objective functions and in such cases **decoupling objectives** become very important. 

Imagine you’re building a system to rank items on users’ newsfeeds. Your original goal is to maximize users’ engagement. You want to achieve this goal through the following three objectives:
- Filter out spam
- Filter out NSFW content
- Rank posts by engagement: how likely users will click on it

However, you quickly learned that optimizing for users’ engagement alone can lead to questionable ethical concerns. Because extreme posts tend to get more engagements, your algorithm learned to prioritize extreme content.

To obtain this goal, you add two new objectives to your original plan:
- Filter out spam
- Filter out NSFW content
- Filter out misinformation
- Rank posts by quality
- Rank posts by engagement: how likely users will click on it

To rank posts by quality, you first need to predict posts’ quality, and you want posts’ predicted quality to be as close to their actual quality as possible. Essentially, you want to minimize quality_loss: the difference between each post’s predicted quality and its true quality.

Similarly, to rank posts by engagement, you first need to predict the number of clicks each post will get. You want to minimize engagement_loss: the difference between each post’s predicted clicks and its actual number of clicks.

One approach is to combine these two losses into one loss and train one model to minimize that loss:

$$ loss = \alpha * quality\_loss + \beta * engagement\_loss $$

A problem with this approach is that each time you tune α and β—for example, if the quality of your users’ newsfeeds goes up but users’ engagement goes down, you might want to decrease α and increase β—you’ll have to retrain your model.

Another approach is to train two different models, each optimizing one loss. So you have two models:
- _quality_model_: Minimizes quality_loss and outputs the predicted quality of each post
- _engagement_model_: Minimizes engagement_loss and outputs the predicted number of clicks of each post

You can combine the models’ outputs and rank posts by their combined scores:

$$ \alpha * quality\_score + \beta * engagement\_score $$

Now you can tweak $\alpha$ and $\beta$ without retraining your models!

# Mind vs Data

One of the most popular debates in ML is which will improve the ML objectives: Data or Model. In other words, if we compare a simple model with lot of data & a complex model with minimal data, which will perform better?

Both the research and industry trends in the recent decades show the success of ML relies more and more on the quality and quantity of data. Models are getting bigger and using more data. Back in 2013, people were getting excited when the One Billion Word Benchmark for Language Modeling was released, which contains 0.8 billion tokens. Six years later, OpenAI’s GPT-2 used a dataset of 10 billion tokens. And another year later, GPT-3 used 500 billion tokens.

When asked how Google Search was doing so well, Peter Norvig, Google’s director of search quality, emphasized the importance of having a large amount of data over intelligent algorithms in their success: “We don’t have better algorithms. We just have more data.”

Dr. Monica Rogati, former VP of data at Jawbone, argued that data lies at the foundation of data science, as shown in the figure below.
![image](https://user-images.githubusercontent.com/20537002/180987771-de572541-3415-4aa3-8851-cb37e34db20b.png)

In the mind-over-data camp, Dr. Judea Pearl emphasizes: “Data is profoundly dumb.” and also claims “ML will not be the same in 3–5 years, and ML folks who continue to follow the current data-centric paradigm will find themselves outdated, if not jobless. Take note.”

There’s also a milder opinion from Professor Christopher Manning, director of the Stanford Artificial Intelligence Laboratory, who argued that huge computation and a massive amount of data with a simple learning algorithm create incredibly bad learners.

Regardless of which camp will prove to be right eventually, no one can deny that data is essential, for now. Even though much of the progress in deep learning in the last decade was fueled by an increasingly large amount of data, more data doesn’t always lead to better performance for your model. More data at lower quality, such as data that is outdated or data with incorrect labels, might even hurt your model’s performance.

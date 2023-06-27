---
title: Understanding Training Data
date: 2022-07-27 15:10:00 +0530
categories: [Book Summary, Designing ML Systems]
tags: [Notes, ML]
math: true
pin: false
image:
  path: https://m.media-amazon.com/images/P/B0B1LGL2SR.01._SCLZZZZZZZ_SX500_.jpg
  lqip: data:image/webp;base64,UklGRpoAAABXRUJQVlA4WAoAAAAQAAAADwAABwAAQUxQSDIAAAARL0AmbZurmr57yyIiqE8oiG0bejIYEQTgqiDA9vqnsUSI6H+oAERp2HZ65qP/VIAWAFZQOCBCAAAA8AEAnQEqEAAIAAVAfCWkAALp8sF8rgRgAP7o9FDvMCkMde9PK7euH5M1m6VWoDXf2FkP3BqV0ZYbO6NA/VFIAAAA
  alt: Designing Machine Learning Systems by Chip Huyen
---

# Reference

[Chapter 4 - Designing Machine Learning Systems by Chip Huyen](https://learning.oreilly.com/library/view/designing-machine-learning/9781098107956/ch04.html)

# Introduction

> Data is messy, complex, unpredictable, and potentially treacherous. If not handled properly, it can easily sink your entire ML operation.

In this post, we will go over techniques to obtain or create good training data. Training data, in this chapter, encompasses all the data used in the developing phase of ML models, including the different splits used for training, validation, and testing (the train, validation, test splits). 

We use the term “training data” instead of “training dataset” because “dataset” denotes a set that is finite and stationary. Data in production is neither finite nor stationary.

> Like other steps in building ML systems, creating training data is an iterative process. As your model evolves through a project lifecycle, your training data will likely also evolve.

# Sampling

In general, we sample training data in one of the two cases:
- When we don't have access to all the possible real-world data
- When it's infeasible to process all the data that we have access to because of time or resource constraints

Understanding different sampling methods and how they are being used in our workflow can, first, help us avoid potential sampling biases, and second, help us choose the methods that improve the efficiency of the data we sample.

## Nonprobability Sampling

Nonprobability sampling is when the selection of data isn’t based on any probability criteria. Here are some of the criteria for nonprobability sampling:

**Convenience sampling**
Samples of data are selected based on their availability. This sampling method is popular because, well, it’s convenient.

**Snowball sampling**
Future samples are selected based on existing samples. For example, to scrape legitimate Twitter accounts without having access to Twitter databases, you start with a small number of accounts, then you scrape all the accounts they follow, and so on.

**Judgment sampling**
Experts decide what samples to include.

**Quota sampling**
You select samples based on quotas for certain slices of data without any randomization. For example, when doing a survey, you might want 100 responses from each of the age groups: under 30 years old, between 30 and 60 years old, and above 60 years old, regardless of the actual age distribution.

The samples selected by nonprobability criteria are not representative of the real-world data and therefore are riddled with selection biases. Unfortunately, in many cases, the selection of data for ML models is still driven by convenience. For example, Language models are often trained not with data that is representative of all possible texts but with data that can be easily collected—Wikipedia, Common Crawl, Reddit.

> Nonprobability sampling can be a quick and easy way to gather your initial data to get your project off the ground. However, for reliable models, you might want to use probability-based sampling.

## Simple Random Sampling

In the simplest form of random sampling, you give all samples in the population equal probabilities of being selected. The advantage of this method is that it’s easy to implement. The drawback is that rare categories of data might not appear in your selection.

## Stratified Sampling

To avoid the drawback of simple random sampling, you can first divide your population into the groups that you care about and sample from each group separately. For example, to sample 1% of data that has two classes, A and B, you can sample 1% of class A and 1% of class B. This way, no matter how rare class A or B is, you’ll ensure that samples from it will be included in the selection. Each group is called a stratum, and this method is called stratified sampling.

## Weighted Sampling

In weighted sampling, each sample is given a weight, which determines the probability of it being selected. This method allows you to leverage domain expertise. For example, if you know that a certain subpopulation of data, such as more recent data, is more valuable to your model and want it to have a higher chance of being selected, you can give it a higher weight.

This also helps with the case when the data you have comes from a different distribution compared to the true data. For example, if in your data, red samples account for 25% and blue samples account for 75%, but you know that in the real world, red and blue have equal probability to happen, you can give red samples weights three times higher than blue samples.

## Reservoir Sampling

Reservoir sampling is a fascinating algorithm that is especially useful when you have to deal with **streaming data**, which is usually what you have in production.

Imagine you have an incoming stream of tweets and you want to sample a certain number, k, of tweets to do analysis or train a model on. We don't know how many tweets there are but we do know they can't all fit in-memory. We want to figure out how to sample such that every tweet is equally likely to be picked and we can stop the algorithm at any point in time.

One solution for this problem is reservoir sampling. The algorithm involves a reservoir, which can be an array, and consists of three steps:
- Put the first $k$ elements into the reservoir.
- For each incoming $nth$ element, generate a random number $i$ such that $1 ≤ i ≤ n$
- If $1 ≤ i ≤ k$ : replace the $ith$ element in the reservoir with the $nth$ element. Else, do nothing.

This means that each incoming $nth$ element has $k \over n$  probability of being in the reservoir. 

# Labeling

> Despite the promise of unsupervised ML, most ML models in production today are supervised, which means that they need labeled data to learn from. The performance of an ML model still depends heavily on the quality and quantity of the labeled data it’s trained on.

Data labeling has gone from being an auxiliary task to being a core function of many ML teams in production.

## Hand Labels

Hand-labeling data can be expensive, especially if subject matter expertise is required. To classify whether a comment is spam, you might be able to find 20 annotators on a crowdsourcing platform and train them in 15 minutes to label your data. However, if you want to label chest X-rays, you’d need to find board-certified radiologists, whose time is limited and expensive.

Hand labeling poses a threat to data privacy. Hand labeling means that someone has to look at your data, which isn’t always possible if your data has strict privacy requirements. And lastly, hand labeling is very slow. Slow labeling leads to slow iteration speed and makes your model less adaptive to changing environments and requirements.

### Label multiplicity

Often, to obtain enough labeled data, companies have to use data from multiple sources and rely on multiple annotators who have different levels of expertise. These different data sources and annotators also have different levels of accuracy. This leads to the problem of label ambiguity or label multiplicity: what to do when there are multiple conflicting labels for a data instance.

To minimize the disagreement among annotators, it’s important to first have a clear problem definition.

### Data Lineage

> Indiscriminately using data from multiple sources, generated with different annotators, without examining their quality can cause your model to fail mysteriously. 

Consider a case when you’ve trained a moderately good model with 100K data samples. Your ML engineers are confident that more data will improve the model performance, so you spend a lot of money to hire annotators to label another million data samples. However, the model performance actually decreases after being trained on the new data. The reason is that the new million samples were crowdsourced to annotators who labeled data with much less accuracy than the original data.

It’s good practice to keep track of the origin of each of your data samples as well as its labels, a technique known as data lineage. Data lineage helps you both flag potential biases in your data and debug your models. 

## Natural Labels

Tasks with natural labels are tasks where the model’s predictions can be automatically evaluated or partially evaluated by the system. Take the example of stock price forecasting for the next two mintues. After the model makes a prediction, we can collect the actual data after two mintues for the ground truth.

Even if your task doesn’t inherently have natural labels, it might be possible to set up your system in a way that allows you to collect some feedback on your model. For example, if you have an anomaly detector system with no labelled data for anomalies, we can get feedback from the user when we flag datapoints as anomalies to mark as a false or true positive.

### Feedback loop length

> For tasks with natural ground truth labels, the time it takes from when a prediction is served until when the feedback on it is provided is the feedback loop length.

Choosing the right window length requires thorough consideration, as it involves the speed and accuracy trade-off. A short window length means that you can capture labels faster, which allows you to use these labels to detect issues with your model and address those issues as soon as possible. However, a short window length also means that you might prematurely label a recommendation as bad before it’s clicked on.

For tasks with long feedback loops, natural labels might not arrive for weeks or even months. Fraud detection is an example of a task with long feedback loops. For a certain period of time after a transaction, users can dispute whether that transaction is fraudulent or not. Labels with long feedback loops are helpful for reporting a model’s performance on quarterly or yearly business reports but not very helpful if you want to detect issues with your models as soon as possible.

## Handling the Lack of Labels

Because of the challenges in acquiring sufficient high-quality labels, many techniques have been developed to address the problems that result.

| Method            | How                                                           | Ground truths required?                                                                                                                                                           |
| ----------------- | ------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Weak supervision  | Leverages (often noisy) heuristics to generate labels         | No, but a small number of labels are recommended to guide the development of heuristics                                                                                           |
| Semi- supervision | Leverages structural assumptions to generate labels           | Yes, a small number of initial labels as seeds to generate more labels                                                                                                            |
| Transfer learning | Leverages models pretrained on another task for your new task | No for zero-shot learning<br>Yes for fine-tuning, though the number of ground truths required is often much smaller than what would be needed if you train the model from scratch |
| Active learning   | Labels data samples that are most useful to your model        | Yes                                                                                                                                                                               |
### Weak Supervision

One approach that has gained popularity is weak supervision. One of the most popular open source tools for weak supervision is Snorkel, developed at the Stanford AI Lab. The insight behind weak supervision is that people rely on heuristics, which can be developed with subject matter expertise, to label data.

Libraries like Snorkel are built around the concept of a labeling function (LF): a function that encodes heuristics. The preceding heuristics can be expressed by the following function:
```python
def labeling_function(note):
   if "pneumonia" in note:
     return "EMERGENT"
```
Because LFs encode heuristics, and heuristics are noisy, labels produced by LFs are noisy. Multiple LFs might apply to the same data examples, and they might give conflicting labels. It’s important to combine, denoise, and reweight all LFs to get a set of most likely to be correct labels.

In theory, you don’t need any hand labels for weak supervision. However, to get a sense of how accurate your LFs are, a small number of hand labels is recommended. With LFs, subject matter expertise can be versioned, reused, and shared. Weak supervision can be especially useful when your data has strict privacy requirements. We can easily scale from 1K to 1M samples. LFs are very adaptive so when changes happen, just reapply LFs!

> In a study with Stanford Medicine, models trained with weakly supervised labels obtained by a single radiologist after eight hours of writing LFs had comparable performance with models trained on data obtained through almost a year of hand labeling. 

If heuristics work so well to label data, why do we need ML models? One reason is that LFs might not cover all data samples, so we can train ML models on data programmatically labeled with LFs and use this trained model to generate predictions for samples that aren’t covered by any LF.

### Semi-Supervision

If weak supervision leverages heuristics to obtain noisy labels, semi-supervision leverages structural assumptions to generate new labels based on a small set of initial labels. Unlike weak supervision, semi-supervision requires an initial set of labels.

A classic semi-supervision method is **self-training**. You start by training a model on your existing set of labeled data and use this model to make predictions for unlabeled samples. 

Another semi-supervision method assumes that data samples that share similar characteristics share the same labels. The similarity might be obvious, such as in the task of classifying the topic of Twitter hashtags. You can start by labeling the hashtag “#AI” as Computer Science.

In most cases, the similarity can only be discovered by more complex methods. For example, you might need to use a clustering method or a k-nearest neighbors algorithm to discover samples that belong to the same cluster.

> One thing to consider when doing semi-supervision with limited data is how much of this limited data should be used to evaluate multiple candidate models and select the best one. 
 
If you use a small amount, the best performing model on this small evaluation set might be the one that overfits the most to this set. On the other hand, if you use a large amount of data for evaluation, the performance boost gained by selecting the best model based on this evaluation set might be less than the boost gained by adding the evaluation set to the limited training set. Many companies overcome this trade-off by using a reasonably large evaluation set to select the best model, then continuing training the champion model on the evaluation set.

### Transfer Learning

Transfer learning refers to the family of methods where a model developed for a task is reused as the starting point for a model on a second task. First, the base model is trained for a base task. The base task is usually a task that has cheap and abundant training data. Language modeling is a great candidate because it doesn’t require labeled data. The trained model can then be used for the task that you’re interested in—a downstream task—such as sentiment analysis, intent detection, or question answering.

> Transfer learning is especially appealing for tasks that don’t have a lot of labeled data. Even for tasks that have a lot of labeled data, using a pretrained model as the starting point can often boost the performance significantly compared to training from scratch.

A trend that has emerged in the last five years is that (usually) the larger the pretrained base model, the better its performance on downstream tasks. Large models are expensive to train. Based on the configuration of GPT-3, it’s estimated that the cost of training this model is in the tens of millions USD.

### Active Learning

The hope here is that ML models can achieve greater accuracy with fewer training labels if they can choose which data samples to learn from. here are multiple variations to this idea but all of them have the stated theme.

> Let the model pick the training data

Active learning involves the following steps:
- Train the model on the labelled data available
- Evaluate the model on all unlabelled samples
- After evaluation, based on a heuristic choose a sample or list of samples to be labelled (manually in most cases)
- Add the newly labelled data to the previous training data and retrain the model
- Repeat this process until certain condition (stopping condition)

The most straightforward metric is uncertainty measurement—label the examples that your model is the least certain about, hoping that they will help your model learn the decision boundary better. 

Another common heuristic is based on disagreement among multiple candidate models. This method is called query-by-committee, an example of an ensemble method. You need a committee of several candidate models, which are usually the same model trained with different sets of hyperparameters or the same model trained on different slices of data. Each model can make one vote for which samples to label next, and it might vote based on how uncertain it is about the prediction. You then label the samples that the committee disagrees on the most.

# Class Imbalance

Class imbalance typically refers to a problem in classification tasks where there is a substantial difference in the number of samples in each class of the training data. For example, in a training dataset for the task of detecting lung cancer from X-ray images, 99.99% of the X-rays might be of normal lungs, and only 0.01% might contain cancerous cells.

Class imbalance can also happen with regression tasks where the labels are continuous. Consider the task of estimating health-care bills.25 Health-care bills are highly skewed—the median bill is low, but the 95th percentile bill is astronomical. When predicting hospital bills, it might be more important to predict accurately the bills at the 95th percentile than the median bills. 

## Challenges of Class Imbalance

ML, especially deep learning, works well in situations when the data distribution is more balanced, and usually not so well when the classes are heavily imbalanced, as illustrated below.
![image](https://user-images.githubusercontent.com/20537002/181303351-3e95081f-0349-450b-9954-6f6eaca03544.png)

Class imbalance can make learning difficult for the following three reasons:
- Class imbalance often means there’s insufficient signal for your model to learn to detect the minority classes
- Class imbalance makes it easier for your model to get stuck in a nonoptimal solution by exploiting a simple heuristic instead of learning anything useful about the underlying pattern of the data
- Class imbalance leads to asymmetric costs of error—the cost of a wrong prediction on a sample of the rare class might be much higher than a wrong prediction on a sample of the majority class.

> In real-world settings, rare events are often more interesting (or more dangerous) than regular events, and many tasks focus on detecting those rare events.

The classical example of tasks with class imbalance is fraud detection. Most credit card transactions are not fraudulent. Outside the cases where class imbalance is inherent in the problem, class imbalance can also be caused by biases during the sampling process. Another cause for class imbalance, though less common, is due to labeling errors.

## Handling Class Imbalance

### Using the right evaluation metrics

The most important thing to do when facing a task with class imbalance is to choose the appropriate evaluation metrics. Wrong metrics will give you the wrong ideas of how your models are doing and, subsequently, won’t be able to help you develop or choose models good enough for your task.

Metrics that help you understand your model’s performance with respect to specific classes would be better choices. Accuracy can still be a good metric if you use it for each class individually. 

F1, precision, and recall are metrics that measure your model’s performance with respect to the positive class in binary classification problems, as they rely on true positive.

|                        | Predicted Positive                         | Predicted Negative                   |
| ---------------------- | ------------------------------------------ | -------------------------------------|
| Positive label         | True Positive (hit)                        | False Negative (type II error, miss) |
| Negative label         | False Positive (type I error, false alarm) | True Negative (correct rejection)    |

$Precision = True Positive / (True Positive + False Positive)$

$Recall = True Positive / (True Positive + False Negative)$

$F1 = 2 × Precision × Recall / (Precision + Recall)$

F1, precision, and recall are asymmetric metrics, which means that their values change depending on which class is considered the positive class.

Many classification problems can be modeled as regression problems. Your model can output a probability, and based on that probability, you classify the sample. For example, if the value is greater than 0.5, it’s a positive label, and if it’s less than or equal to 0.5, it’s a negative label. This means that you can tune the threshold to increase the _true positive rate_ (also known as **recall**) while decreasing the _false positive rate_ (also known as the _probability of false alarm_), and vice versa. We can plot the true positive rate against the false positive rate for different thresholds. This plot is known as the ROC curve (receiver operating characteristics). When your model is perfect, the recall is 1.0, and the curve is just a line at the top.

The area under the curve (AUC) measures the area under the ROC curve. Since the closer to the perfect line the better, the larger this area the better.

![image](https://user-images.githubusercontent.com/20537002/181307492-c5369bce-d6bd-4649-9851-955f0708e6f9.png)

Like F1 and recall, the ROC curve focuses only on the positive class and doesn’t show how well your model does on the negative class. Davis and Goadrich suggested that we should plot precision against recall instead, in what they termed the Precision-Recall Curve. They argued that this curve gives a more informative picture of an algorithm’s performance on tasks with heavy class imbalance.

### Data-level methods: Resampling

Data-level methods modify the distribution of the training data to reduce the level of imbalance to make it easier for the model to learn. A common family of techniques is resampling.

Resampling includes oversampling, adding more instances from the minority classes, and undersampling, removing instances of the majority classes. The simplest way to undersample is to randomly remove instances from the majority class, whereas the simplest way to oversample is to randomly make copies of the minority class until you have a ratio that you’re happy with.

Undersampling runs the risk of losing important data from removing data. Oversampling runs the risk of overfitting on training data, especially if the added copies of the minority class are replicas of existing data. Many sophisticated sampling techniques have been developed to mitigate these risks.

One such technique is two-phase learning. You first train your model on the resampled data. This resampled data can be achieved by randomly undersampling large classes until each class has only N instances. You then fine-tune your model on the original data.

Another technique is dynamic sampling: oversample the low-performing classes and undersample the high-performing classes during the training process. This method aims to show the model less of what it has already learned and more of what it has not.

### Algorithm-level methods

If data-level methods mitigate the challenge of class imbalance by altering the distribution of your training data, algorithm-level methods keep the training data distribution intact but alter the algorithm to make it more robust to class imbalance.

Because the loss function (or the cost function) guides the learning process, many algorithm-level methods involve adjustment to the loss function. The key idea is that if there are two instances, x1 and x2, and the loss resulting from making the wrong prediction on x1 is higher than x2, the model will prioritize making the correct prediction on x1 over making the correct prediction on x2. By giving the training instances we care about higher weight, we can make the model focus more on learning these instances.

#### Cost-sensitive learning

Back in 2001, based on the insight that misclassification of different classes incurs different costs, Elkan proposed cost-sensitive learning in which the individual loss function is modified to take into account this varying cost.

#### Class-balanced loss

What might happen with a model trained on an imbalanced dataset is that it’ll bias toward majority classes and make wrong predictions on minority classes. What if we punish the model for making wrong predictions on minority classes to correct this bias?

In its vanilla form, we can make the weight of each class inversely proportional to the number of samples in that class, so that the rarer classes have higher weights.

#### Focal loss

In our data, some examples are easier to classify than others, and our model might learn to classify them quickly. We want to incentivize our model to focus on learning the samples it still has difficulty classifying. What if we adjust the loss so that if a sample has a lower probability of being right, it’ll have a higher weight? This is exactly what focal loss does.




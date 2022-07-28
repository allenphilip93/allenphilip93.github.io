---
title: Understanding Feature Engineering
date: 2022-07-27 21:40:00 +0530
categories: [ML]
tags: [Notes, Learning]
math: true
pin: false
---

# Reference

[Chapter 5 - Designing Machine Learning Systems by Chip Huyen](https://learning.oreilly.com/library/view/designing-machine-learning/9781098107956/ch05.html)

# Introduction

> In 2014, the paper “[Practical Lessons from Predicting Clicks on Ads at Facebook](https://scontent.fbom26-1.fna.fbcdn.net/v/t39.8562-6/240842589_204052295113548_74168590424110542_n.pdf?_nc_cat=109&ccb=1-7&_nc_sid=ad8a9d&_nc_ohc=576ichZP9XQAX-ZouMc&_nc_ht=scontent.fbom26-1.fna&oh=00_AT9CzcC5ZbtRQn79CwpV8ZVPuhcQJvj1x3OcLIA5Md8XbA&oe=62E6C6CA)” claimed that having the right features is the most important thing in developing their ML models.

 State-of-the-art model architectures can still perform poorly if they don’t use a good set of features. Due to its importance, a large part of many ML engineering and data science jobs is to come up with new useful features. In this post, we will go over common techniques and important considerations with respect to feature engineering.

# Learned Features Versus Engineered Features 

> Do we really have to worry about feature engineering? Doesn’t deep learning promise us that we no longer have to engineer features?

That is right. The promise of deep learning is that we won’t have to handcraft features. For this reason, deep learning is sometimes called feature learning. However, we’re still far from the point where all features can be automated. This is not to mention that, as of this writing, the majority of ML applications in production aren’t deep learning.

Imagine that you want to build a sentiment analysis classifier to classify whether a comment is spam or not. Before deep learning, when given a piece of text, you would have to manually apply classical text processing techniques such as lemmatization, expanding contractions, removing punctuation, and lowercasing everything. After that, you might want to split your text into n-grams with n values of your choice as shown below.

![image](https://user-images.githubusercontent.com/20537002/181421769-cd2d2b51-a047-4ce8-8d0c-9f076f0146e4.png)

Once you’ve generated n-grams for your training data, you can create a vocabulary that maps each n-gram to an index. Then you can convert each post into a vector based on its n-grams’ indices. For example, if we have a vocabulary of seven n-grams as shown in table below.

| I | like | good | food | I like | good food | like food |
| - | ---- | ---- | ---- | ------ | --------- | --------- |
| 0 | 1    | 2    | 3    | 4      | 5         | 6         |

Each element corresponds to the number of times the n-gram at that index appears in the post. “I like food” will be encoded as the vector [1, 1, 0, 1, 1, 0, 1]. This vector can then be used as an input into an ML model.

Feature engineering requires knowledge of domain-specific techniques—in this case, the domain is natural language processing (NLP) and the native language of the text. It tends to be an iterative process, which can be brittle.

However, much of this pain has been alleviated since the rise of deep learning. Instead of having to worry about lemmatization, punctuation, or stopword removal, you can just split your raw text into words (i.e., tokenization), create a vocabulary out of those words, and convert each of your words into one-shot vectors using this vocabulary. Your model will hopefully learn to extract useful features from this. In this new method, much of feature engineering for text has been automated.

However, an ML system will likely need data beyond just text and images. For example, when detecting whether a comment is spam or not, on top of the text in the comment itself like:
- How many upvotes/downvotes does it have?
- When was this account created, how often do they post, and how many upvotes/downvotes do they have?
- How many views does it have? Popular threads tend to attract more spam.

> The process of choosing what information to use and how to extract this information into a format usable by your ML models is feature engineering.

# Common Feature Engineering Operations

Because of the importance and the ubiquity of feature engineering in ML projects, there have been many techniques developed to streamline the process.

## Handling Missing Values

> It's important to keep in mind that not all types of missing values are equal

There are three types of missing values:

**Missing not at random (MNAR)**

This is when the reason a value is missing is because of the true value itself. For example, consider a dataset with income of tenants in a region. Some of the tenants may choose to not disclose their income and hence they may be null.

**Missing at random (MAR)**

This is when the reason a value is missing is not due to the value itself, but due to another observed variable. In this example, we might notice that age values are often missing for respondents of the gender “A,” which might be because the people of gender A in this survey don’t like disclosing their age.

**Missing completely at random (MCAR)**

This is when there’s no pattern in when the value is missing. In this example, we might think that the missing values for the column “Job” might be completely random, not because of the job itself and not because of any other variable. People just forget to fill in that value sometimes for no particular reason. However, this type of missing is very rare.

### Deletion

One most preferred way to handle missing values is deletion, not because it's a better method but because it's easier to do.

One way to delete is **column deletion**: if a variable has too many missing values, just remove that variable.

Another way to delete is **row deletion**: if a sample has missing value(s), just remove that sample. This method can work when the missing values are completely at random (MCAR) and the number of examples with missing values is small, such as less than 0.1%.

However, removing rows of data can also remove important information that your model needs to make predictions, especially if the missing values are not at random (MNAR). On top of that, removing rows of data can create biases in your model, especially if the missing values are at random (MAR).

### Imputation

If you don’t want to delete missing values, you will have to impute them, which means “fill them with certain values.” Deciding which “certain values” to use is the hard part.

One common practice is to fill in missing values with their defaults. For example, if the job is missing, you might fill it with an empty string “”. Another common practice is to fill in missing values with the mean, median, or mode (the most common value). 

In general, you want to avoid filling missing values with possible values, such as filling the missing number of children with 0—0 is a possible value for the number of children. It makes it hard to distinguish between people whose information is missing and people who don’t have children.

## Scaling

When we input these two variables into an ML model, it won’t understand that 150,000 and 40 represent different things. It will just see them both as numbers, and because the number 150,000 is much bigger than 40, it might give it more importance, regardless of which variable is actually more useful for generating predictions.

> Before inputting features into models, it’s important to scale them to be similar ranges. This process is called feature scaling.

An intuitive way to scale your features is to get them to be in the range [0, 1]. The goal of normalization is to change the values of numeric columns in the dataset to a common scale, without distorting differences in the ranges of values. For machine learning, every dataset does not require normalization. It is required only when features have different ranges.

If you think that your variables might follow a normal distribution, it might be helpful to normalize them so that they have zero mean and unit variance. This process is called standardization. Standardization assumes that your data has a Gaussian (bell curve) distribution. This does not strictly have to be true, but the technique is more effective if your attribute distribution is Gaussian.

In practice, ML models tend to struggle with features that follow a skewed distribution. To help mitigate the skewness, a technique commonly used is log transformation: apply the log function to your feature which can make your data less skewed. While this technique can yield performance gain in many cases, it doesn’t work for all cases, and you should be wary of the analysis performed on log-transformed data instead of the original data.

There are two important things to note about scaling. 

> One is that it’s a common source of data leakage. Another is that it often requires global statistics—you have to look at the entire or a subset of training data to calculate its min, max, or mean. 
 
During inference, you reuse the statistics you had obtained during training to scale new data. If the new data has changed significantly compared to the training, these statistics won’t be very useful. Therefore, it’s important to retrain your model often to account for these changes.

## Encoding Categorical Features

Imagine you’re building a recommender system to predict what products users might want to buy from Amazon. One of the features you want to use is the product brand. When looking at Amazon’s historical data, you realize that there are a lot of brands. Even back in 2019, there were already over two million brands on Amazon!

You encode each brand as a number, so now you have two million numbers, from 0 to 1,999,999, corresponding to two million brands. In production, your model crashes because it encounters a brand it hasn’t seen before and therefore can’t encode. New brands join Amazon all the time. 

To address this, you create a category UNKNOWN with the value of 2,000,000 to catch all the brands your model hasn’t seen during training. Your model doesn’t crash anymore, but your sellers complain that their new brands are not getting any traffic. It’s because your model didn’t see the category UNKNOWN in the train set, so it just doesn’t recommend any product of the UNKNOWN brand. 

You fix this by encoding only the top 99% most popular brands and encode the bottom 1% brand as UNKNOWN. This way, at least your model knows how to deal with UNKNOWN brands. Over the last hour, 20 new brands joined your site; some of them are new luxury brands, some of them are sketchy knockoff brands, some of them are established brands. However, your model treats them all the same way it treats unpopular brands in the training data.

> Finding a way to solve this problem turns out to be surprisingly difficult.

One solution to this problem is the hashing trick, popularized by the package Vowpal Wabbit developed at Microsoft.7 The gist of this trick is that you use a hash function to generate a hashed value of each category. The hashed value will become the index of that category. Because you can specify the hash space, you can fix the number of encoded values for a feature in advance, without having to know how many categories there will be. For example, if you choose a hash space of 18 bits, which corresponds to 218 = 262,144 possible hashed values, all the categories, even the ones that your model has never seen before, will be encoded by an index between 0 and 262,143. One problem with hashed functions is **collision** but in research done by Booking.com, even for 50% colliding features, the performance loss is less than 0.5%!

## Feature Crossing

Feature crossing is the technique to combine two or more features to generate new features. This technique is useful to model the nonlinear relationships between features. 

For example, for the task of predicting whether someone will want to buy a house in the next 12 months, you suspect that there might be a nonlinear relationship between marital status and number of children, so you combine them to create a new feature “marriage and children”.

Because feature crossing helps model nonlinear relationships between variables, it’s essential for models that can’t learn or are bad at learning nonlinear relationships, such as linear regression, logistic regression, and tree-based models. But a caveat of feature crossing is that it can make your feature space blow up. Imagine feature A & B have 100 possible values; after we cross them we will end up with 100 x 100 features!

# Data Leakage



---
title: Understanding Feature Engineering
date: 2022-07-27 21:40:00 +0530
categories: [Book Summary, Designing ML Systems]
tags: [Notes, ML]
math: true
pin: false
image:
  path: https://m.media-amazon.com/images/P/B0B1LGL2SR.01._SCLZZZZZZZ_SX500_.jpg
  lqip: data:image/webp;base64,UklGRpoAAABXRUJQVlA4WAoAAAAQAAAADwAABwAAQUxQSDIAAAARL0AmbZurmr57yyIiqE8oiG0bejIYEQTgqiDA9vqnsUSI6H+oAERp2HZ65qP/VIAWAFZQOCBCAAAA8AEAnQEqEAAIAAVAfCWkAALp8sF8rgRgAP7o9FDvMCkMde9PK7euH5M1m6VWoDXf2FkP3BqV0ZYbO6NA/VFIAAAA
  alt: Designing Machine Learning Systems by Chip Huyen
---

## Reference

[Chapter 5 - Designing Machine Learning Systems by Chip Huyen](https://learning.oreilly.com/library/view/designing-machine-learning/9781098107956/ch05.html)

## Introduction

> In 2014, the paper “[Practical Lessons from Predicting Clicks on Ads at Facebook](https://scontent.fbom26-1.fna.fbcdn.net/v/t39.8562-6/240842589_204052295113548_74168590424110542_n.pdf?_nc_cat=109&ccb=1-7&_nc_sid=ad8a9d&_nc_ohc=576ichZP9XQAX-ZouMc&_nc_ht=scontent.fbom26-1.fna&oh=00_AT9CzcC5ZbtRQn79CwpV8ZVPuhcQJvj1x3OcLIA5Md8XbA&oe=62E6C6CA)” claimed that having the right features is the most important thing in developing their ML models.

 State-of-the-art model architectures can still perform poorly if they don’t use a good set of features. Due to its importance, a large part of many ML engineering and data science jobs is to come up with new useful features. In this post, we will go over common techniques and important considerations with respect to feature engineering.

## Learned Features Versus Engineered Features 

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

## Common Feature Engineering Operations

Because of the importance and the ubiquity of feature engineering in ML projects, there have been many techniques developed to streamline the process.

### Handling Missing Values

> It's important to keep in mind that not all types of missing values are equal

There are three types of missing values:

**Missing not at random (MNAR)**

This is when the reason a value is missing is because of the true value itself. For example, consider a dataset with income of tenants in a region. Some of the tenants may choose to not disclose their income and hence they may be null.

**Missing at random (MAR)**

This is when the reason a value is missing is not due to the value itself, but due to another observed variable. In this example, we might notice that age values are often missing for respondents of the gender “A,” which might be because the people of gender A in this survey don’t like disclosing their age.

**Missing completely at random (MCAR)**

This is when there’s no pattern in when the value is missing. In this example, we might think that the missing values for the column “Job” might be completely random, not because of the job itself and not because of any other variable. People just forget to fill in that value sometimes for no particular reason. However, this type of missing is very rare.

#### Deletion

One most preferred way to handle missing values is deletion, not because it's a better method but because it's easier to do.

One way to delete is **column deletion**: if a variable has too many missing values, just remove that variable.

Another way to delete is **row deletion**: if a sample has missing value(s), just remove that sample. This method can work when the missing values are completely at random (MCAR) and the number of examples with missing values is small, such as less than 0.1%.

However, removing rows of data can also remove important information that your model needs to make predictions, especially if the missing values are not at random (MNAR). On top of that, removing rows of data can create biases in your model, especially if the missing values are at random (MAR).

#### Imputation

If you don’t want to delete missing values, you will have to impute them, which means “fill them with certain values.” Deciding which “certain values” to use is the hard part.

One common practice is to fill in missing values with their defaults. For example, if the job is missing, you might fill it with an empty string “”. Another common practice is to fill in missing values with the mean, median, or mode (the most common value). 

In general, you want to avoid filling missing values with possible values, such as filling the missing number of children with 0—0 is a possible value for the number of children. It makes it hard to distinguish between people whose information is missing and people who don’t have children.

### Scaling

When we input these two variables into an ML model, it won’t understand that 150,000 and 40 represent different things. It will just see them both as numbers, and because the number 150,000 is much bigger than 40, it might give it more importance, regardless of which variable is actually more useful for generating predictions.

> Before inputting features into models, it’s important to scale them to be similar ranges. This process is called feature scaling.

An intuitive way to scale your features is to get them to be in the range [0, 1]. The goal of normalization is to change the values of numeric columns in the dataset to a common scale, without distorting differences in the ranges of values. For machine learning, every dataset does not require normalization. It is required only when features have different ranges.

If you think that your variables might follow a normal distribution, it might be helpful to normalize them so that they have zero mean and unit variance. This process is called standardization. Standardization assumes that your data has a Gaussian (bell curve) distribution. This does not strictly have to be true, but the technique is more effective if your attribute distribution is Gaussian.

In practice, ML models tend to struggle with features that follow a skewed distribution. To help mitigate the skewness, a technique commonly used is log transformation: apply the log function to your feature which can make your data less skewed. While this technique can yield performance gain in many cases, it doesn’t work for all cases, and you should be wary of the analysis performed on log-transformed data instead of the original data.

There are two important things to note about scaling. 

> One is that it’s a common source of data leakage. Another is that it often requires global statistics—you have to look at the entire or a subset of training data to calculate its min, max, or mean. 
 
During inference, you reuse the statistics you had obtained during training to scale new data. If the new data has changed significantly compared to the training, these statistics won’t be very useful. Therefore, it’s important to retrain your model often to account for these changes.

### Encoding Categorical Features

Imagine you’re building a recommender system to predict what products users might want to buy from Amazon. One of the features you want to use is the product brand. When looking at Amazon’s historical data, you realize that there are a lot of brands. Even back in 2019, there were already over two million brands on Amazon!

You encode each brand as a number, so now you have two million numbers, from 0 to 1,999,999, corresponding to two million brands. In production, your model crashes because it encounters a brand it hasn’t seen before and therefore can’t encode. New brands join Amazon all the time. 

To address this, you create a category UNKNOWN with the value of 2,000,000 to catch all the brands your model hasn’t seen during training. Your model doesn’t crash anymore, but your sellers complain that their new brands are not getting any traffic. It’s because your model didn’t see the category UNKNOWN in the train set, so it just doesn’t recommend any product of the UNKNOWN brand. 

You fix this by encoding only the top 99% most popular brands and encode the bottom 1% brand as UNKNOWN. This way, at least your model knows how to deal with UNKNOWN brands. Over the last hour, 20 new brands joined your site; some of them are new luxury brands, some of them are sketchy knockoff brands, some of them are established brands. However, your model treats them all the same way it treats unpopular brands in the training data.

> Finding a way to solve this problem turns out to be surprisingly difficult.

One solution to this problem is the hashing trick, popularized by the package Vowpal Wabbit developed at Microsoft.7 The gist of this trick is that you use a hash function to generate a hashed value of each category. The hashed value will become the index of that category. Because you can specify the hash space, you can fix the number of encoded values for a feature in advance, without having to know how many categories there will be. For example, if you choose a hash space of 18 bits, which corresponds to 218 = 262,144 possible hashed values, all the categories, even the ones that your model has never seen before, will be encoded by an index between 0 and 262,143. One problem with hashed functions is **collision** but in research done by Booking.com, even for 50% colliding features, the performance loss is less than 0.5%!

### Feature Crossing

Feature crossing is the technique to combine two or more features to generate new features. This technique is useful to model the nonlinear relationships between features. 

For example, for the task of predicting whether someone will want to buy a house in the next 12 months, you suspect that there might be a nonlinear relationship between marital status and number of children, so you combine them to create a new feature “marriage and children”.

Because feature crossing helps model nonlinear relationships between variables, it’s essential for models that can’t learn or are bad at learning nonlinear relationships, such as linear regression, logistic regression, and tree-based models. But a caveat of feature crossing is that it can make your feature space blow up. Imagine feature A & B have 100 possible values; after we cross them we will end up with 100 x 100 features!

## Data Leakage

In one example, researchers trained their model on a mix of scans taken when patients were lying down and standing up. “Because patients scanned while lying down were more likely to be seriously ill, the model learned to predict serious covid risk from a person’s position.”

In some other cases, models were “found to be picking up on the text font that certain hospitals used to label the scans. As a result, fonts from hospitals with more serious caseloads became predictors of covid risk.”

> Data leakage refers to the phenomenon when a form of the label “leaks” into the set of features used for making predictions, and this same information is not available during inference.

Data leakage is challenging because often the leakage is nonobvious. It’s dangerous because it can cause your models to fail in an unexpected and spectacular way, even after extensive evaluation and testing.

### Common Causes for Data Leakage

#### Splitting time-correlated data randomly instead of by time

In many cases, data is time-correlated, which means that the time the data is generated affects its label distribution. Sometimes, the correlation is obvious, as in the case of stock prices. If 90% of the tech stocks go down today, it’s very likely the other 10% of the tech stocks go down too. When building models to predict the future stock prices, you want to split your training data by time, such as training your model on data from the first six days and evaluating it on data from the seventh day. If you randomly split your data, prices from the seventh day will be included in your train split and leak into your model the condition of the market on that day.

![image](https://user-images.githubusercontent.com/20537002/181435467-02c0731f-2429-4262-9616-4e398bda1436.png)

#### Scaling before splitting

One common mistake is to use the entire training data to generate global statistics before splitting it into different splits, leaking the mean and variance of the test samples into the training process, allowing a model to adjust its predictions for the test samples.

To avoid this type of leakage, always split your data first before scaling, then use the statistics from the train split to scale all the splits.

#### Filling in missing data with statistics from the test split

One common way to handle the missing values of a feature is to fill (input) them with the mean or median of all values present. Leakage might occur if the mean or median is calculated using entire data instead of just the train split.

#### Poor handling of data duplication before splitting

If you have duplicates or near-duplicates in your data, failing to remove them before splitting your data might cause the same samples to appear in both train and validation/test splits. Data duplication can result from data collection or merging of different data sources.

#### Leakage from data generation process

The example earlier about how information on whether a CT scan shows signs of lung cancer is leaked via the scan machine is an example of this type of leakage. Detecting this type of data leakage requires a deep understanding of the way data is collected. 

### Detecting Data Leakage

> Data leakage can happen during many steps, from generating, collecting, sampling, splitting, and processing data to feature engineering. 

Measure the predictive power of each feature or a set of features with respect to the target variable (label). If a feature has unusually high correlation, investigate how this feature is generated and whether the correlation makes sense.

Do ablation studies to measure how important a feature or a set of features is to your model. If removing a feature causes the model’s performance to deteriorate significantly, investigate why that feature is so important.

Keep an eye out for new features added to your model. If adding a new feature significantly improves your model’s performance, either that feature is really good or that feature just contains leaked information about labels.

Lastly be very careful every time you look at the test split. If you use the test split in any way other than to report a model’s final performance, whether to come up with ideas for new features or to tune hyperparameters, you risk leaking information from the future into your training process.

## Engineering Good Features

> Generally, adding more features leads to better model performance.

However, having too many features can be bad both during training and serving your model for the following reasons:
- The more features you have, the more opportunities there are for data leakage.
- Too many features can cause overfitting.
- Too many features can increase memory required to serve a model driving the expenses up.
- Too many features can increase inference latency when doing online prediction, especially if you need to extract these features from raw data.
- Useless features become technical debts. 

Though we can employ regularization techniques like L1 regularization to reduce the useless feature weights to 0, in practice, removing such features helps the model run faster and prioritize good features.

There are two factors you might want to consider when evaluating whether a feature is good for a model: importance to the model and generalization to unseen data.

### Feature Importance

There are many different methods for measuring a feature’s importance like the:
- In-build feature importance feature in xgboost
- SHAP values for a model-agnostic method
- InterpretML opensource package

The exact algorithm for feature importance measurement is complex, but intuitively, a feature’s importance to a model is measured by how much that model’s performance deteriorates if that feature or a set of features containing that feature is removed from the model. 

> SHAP is great because it not only measures a feature’s importance to an entire model, it also measures each feature’s contribution to a model’s specific prediction. 

Often, a small number of features accounts for a large portion of your model’s feature importance.Not only good for choosing the right features, feature importance techniques are also great for interpretability as they help you understand how your models work under the hood.

### Feature Generalization

Since the goal of an ML model is to make correct predictions on unseen data, features used for the model should generalize to unseen data. Measuring feature generalization is a lot less scientific than measuring feature importance, and it requires both intuition and subject matter expertise on top of statistical knowledge.

Coverage is the percentage of the samples that has values for this feature in the data—so the fewer values that are missing, the higher the coverage. A rough rule of thumb is that if this feature appears in a very small percentage of your data, it’s not going to be very generalizable.

Also for the feature values that are present, you might want to look into their distribution. If the set of values that appears in the seen data (such as the train split) has no overlap with the set of values that appears in the unseen data (such as the test split), this feature might even hurt your model’s performance.

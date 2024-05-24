---
title: Machine Learning Systems
date: 2022-07-22 12:13:00 +0530
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

[Chapter 1 - Designing Machine Learning Systems by Chip Huyen](https://learning.oreilly.com/library/view/designing-machine-learning/9781098107956/ch01.html)

## Motivation

> In November 2016, Google announced that it had incorporated its multilingual neural machine translation system into Google Translate, marking one of the first success stories of deep artificial neural networks in production at scale.1 According to Google, with this update, the quality of translation improved more in a single leap than they had seen in the previous 10 years combined.

In just five years, ML has found its way into almost every aspect of our lives: how we access information, how we communicate, how we work, how we find love. The spread of ML has been so rapid that it’s already hard to imagine life without it. 

Machine learning systems sit at the heart of all these solutions and has become so prevelant in the last few years. Hence it makes sense to understand and learn how to be build ML systems. 

## When to use ML

As tempting as it maybe to quickly using the state-of-the-art ML algorithms, we should exercise caution and ensure that machine learning is necessary and adds value in solving the business problem.

Before understanding when to use ML let's understand what a ML solution does:

> Machine learning is an approach to (1) learn (2) complex patterns from (3) existing data and use these patterns to make (4) predictions on (5) unseen data.

- Learn: the system has the capacity to learn
- Complex patterns: there are patterns to learn, and they are complex
- Existing data: data is available, or it’s possible to collect data
- Predictions: it’s a predictive problem
- Unseen data: unseen data shares patterns with the training data

Though not essential as the above reasons to use ML, we can extend few reasons as good-to-have to reinforce our case for a ML solution
- It’s repetitive
- The cost of wrong predictions is cheap
- It’s at scale
- The patterns are constantly changing

While keep the above reasons to use ML are important, it also useful to keep in mind when NOT to use ML solutions:
- It’s unethical
- Simpler solutions do the trick
- It’s not cost-effective

## Machine Learning Use Cases
According to Algorithmia’s 2020 state of enterprise machine learning survey, ML applications in enterprises are diverse, serving both internal use cases (reducing costs, generating customer insights and intelligence, internal processing automation) and external use cases (improving customer experience, retaining customers, interacting with customers) as shown in the below figure.

<img width="647" alt="image" src="https://user-images.githubusercontent.com/20537002/180387908-066e738f-5537-4602-8fbc-96f6114d4ce8.png">

Fraud detection is among the oldest applications of ML in the enterprise world. If your product or service involves transactions of any value, it’ll be susceptible to fraud. By leveraging ML solutions for anomaly detection, you can have systems that learn from historical fraud transactions and predict whether a future transaction is fraudulent.

Deciding how much to charge for your product or service is probably one of the hardest business decisions; why not let ML do it for you? Price optimization is the process of estimating a price at a certain time period to maximize a defined objective function, such as the company’s margin, revenue, or growth rate. ML-based pricing optimization is most suitable for cases with a large number of transactions where demand fluctuates and consumers are willing to pay a dynamic price—for example, internet ads, flight tickets, accommodation bookings, ride-sharing, and events.

## Understanding Machine Learning Systems

### Machine Learning in Research Versus in Production

As with the more traditional software development, most people with ML expertise have gained it through academia: taking courses, doing research, reading academic papers. It might be a steep learning curve to understand the challenges of deploying ML systems in the wild and navigate an overwhelming set of solutions to these challenges. ML in production is very different from ML in research.

|   | <br>Research           | Production                                                                                                       |
| - | --------------------- | ---------------------------------------------------------------------------------------------------------------- |
| Requirements           | State-of-the-art model performance on benchmark datasets                                                         | Different stakeholders have different requirements |
| Computational priority | Fast training, high throughput                                                                                   | Fast inference, low latency |
| Data                   | Static | Constantly shifting |
| Fairness               | Often not a focus                                                                                                | Must be considered |
| Interpretability       | Often not a focus                                                                                                | Must be considered |

### Different stakeholders and requirements

There are many stakeholders involved in bringing an ML system into production. Each stakeholder has their own requirements. Having different, often conflicting, requirements can make it difficult to design, develop, and select an ML model that satisfies all the requirements.
Consider a mobile app that recommends restaurants to users. The app makes money by charging restaurants a 10% service fee on each order. This means that expensive orders give the app more money than cheap orders. The project involves ML engineers, salespeople, product managers, infrastructure engineers, and a manager:

**ML engineers**

Want a model that recommends restaurants that users will most likely order from, and they believe they can do so by using a more complex model with more data.

**Sales team**

Wants a model that recommends the more expensive restaurants since these restaurants bring in more service fees.

**Product team**

Notices that every increase in latency leads to a drop in orders through the service, so they want a model that can return the recommended restaurants in less than 100 milliseconds.

**ML platform team**

As the traffic grows, this team has been woken up in the middle of the night because of problems with scaling their existing system, so they want to hold off on model updates to prioritize improving the ML platform.

**Manager**

Wants to maximize the margin, and one way to achieve this might be to let go of the ML team.

“Recommending the restaurants that users are most likely to click on” and “recommending the restaurants that will bring in the most money for the app” are two different objectives. Spoiler: we’ll develop one model for each objective and combine their predictions.

When developing an ML project, it’s important for ML engineers to understand requirements from all stakeholders involved and how strict these requirements are.

Production having different requirements from research is one of the reasons why successful research projects might not always be used in production. For example, ensembling is a technique popular among the winners of many ML competitions, including the famed $1 million Netflix Prize, and yet it’s not widely used in production. Ensembling combines “multiple learning algorithms to obtain better predictive performance than could be obtained from any of the constituent learning algorithms alone.” While it can give your ML system a small performance improvement, ensembling tends to make a system too complex to be useful in production, e.g., slower to make predictions or harder to interpret the results.

### Hidden Technical Debt in ML Systems

Often at times, it easy to overlook the technical effort that is required to support a ML solution. [Hidden Technical Debt in ML Systems](https://proceedings.neurips.cc/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf) paper highlights this beautifully and is worth a read.

![image](https://user-images.githubusercontent.com/20537002/180414660-c43e9f2e-4f8f-449c-b4f1-20262da1049b.png)



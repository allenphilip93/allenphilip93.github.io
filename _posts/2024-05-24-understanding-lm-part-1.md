---
title: Understanding LMs - Overview
date: 2024-05-24 11:22:00 +0530
categories: [Learning, Large Models]
tags: [Notes, ML, GenAI]
math: false
pin: false
image:
  path: https://www.cohesity.com/wp-content/new_media/2023/05/blog-RAG_Hero-925x440-1.png
  lqip: data:image/webp;base64,UklGRpoAAABXRUJQVlA4WAoAAAAQAAAADwAABwAAQUxQSDIAAAARL0AmbZurmr57yyIiqE8oiG0bejIYEQTgqiDA9vqnsUSI6H+oAERp2HZ65qP/VIAWAFZQOCBCAAAA8AEAnQEqEAAIAAVAfCWkAALp8sF8rgRgAP7o9FDvMCkMde9PK7euH5M1m6VWoDXf2FkP3BqV0ZYbO6NA/VFIAAAA
  alt: Understanding LMs - Overview
---

## What's in this LM Series?

The purpose of this series if two-fold:
- Serve as a template for people who are on the LLM/LVM/LAM learning journey
- Capture my learnings and serve as notes for me to go back in the future

This will be a summarization of various blogs, videos, lectures, papers & my interpretations of them. To this end, I attempt to break down each section into the follow:
- First I aim to first gather a high-level understanding
- Then work my way to a intuitive understanding of the internals & concepts
- Post this I'll attempt an implementation using huggingface
- Lastly, I'll try to build the whole thing from scratch using Pytorch, NumPy etc

There are plenty of resources out there that does all of these sections very well, so I'll attempt to collate the information in way I can understand the same and hopefully it'll be useful for others as well.

## Pre-requisites

Most of the high-level understanding & building intuition on the LLM concepts should be accessible to all. Parts of the LLM Series works on the assumption that you have a working understanding of ML Fundamentals esp neural networks & at an intermediate level in Python & PyTorch. If you feel you're good on these fronts please skip ahead to the next section, if you can use this section to serve as a starter/refresher on these concepts.

### ML Fundamentals

But it certainly (though not mandatory) helps to have an understanding of Machine Learning Fundamentals, so if you want to get up to speed on the ML basic you can use these references:
- [Coursera - ML Specialization by Andrew NG](https://www.coursera.org/specializations/machine-learning-introduction)
	- One of the most popular courses to get started with ML and personally it helped me a lot
	- The course may seem to be a bit theoretical but if you persist you'll get a strong intuitive understanding of ML concepts
- [ML Crash Course from Google](https://developers.google.com/machine-learning/crash-course/ml-intro)
	- This should give a quick (but not thorough) introduction to the world of ML with a lot of hand-on examples with TensorFlow
	- Good options if you're short on time and want to quickly get started
- [Hand-On Machine Learning - Book](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781492032632/)
	- I'm a few chapters and so far I've really like it. Like the book says it super hand-on and I found myself finding nice ways to use Pandas, NumPy and Scikit-Learn
	- Book is fairly comprehensive as well going upto RL & Generative Modelling but don't expect too much detail beyond the CNN section but it can give a good overview of the concepts

Also to get really behind on the Math behind the ML models, having a quick refresher on Linear Algebra, Matrix manipulations, Probabilities etc helps. One favourite reference of mine if the playlist from `3Blue1Brown` 
- [Essense of Linear Algebra](https://www.youtube.com/watch?v=fNk_zzaMoSs&list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab&ab_channel=3Blue1Brown)
	- Excellent way to visualize and understand fundamental Linear Algebra concepts. Do check out!

### Python
Python is almost synonymous with ML these days. It's the most popular language for using ML models, LLM, writing the pipelines for such workflows in production so it's quite essential to have a decent hands-on experience with Python. Again it's not required if you just need a high-level understanding but atleast personally I tend to understand things better when I get my hands dirty so to that end having Python knowledge helps!

You can use some of the references below to get started with Python:
- [Official Python Tutorial](https://docs.python.org/3/tutorial/index.html)
	- Where better to start than the official python tutorial.
	- It's a bit extensive so it may take awhile to go through.
- [LearnPython.org](https://www.learnpython.org/)
	- Super hand-on way to get started with Python.
- [Machine Learning Basics - Github](https://github.com/SamBelkacem/Machine-Learning-Basics/tree/main)
	- This repo all has the resources to get you started on your ML journey with Python.

### PyTorch

PyTorch and TensorFlow are two of the most prominent deep learning frameworks used in developing and training large language models (LLMs). PyTorch is renowned for its intuitive and pythonic interface, which makes it easier for researchers and developers to write and understand code. This is particularly important in the context of LLMs, where experimentation and iteration are frequent.

You can get started with PyTorch with the resources here:
1. **[PyTorch Official Documentation](https://pytorch.org/tutorials/beginner/basics/intro.html)**: The official documentation is comprehensive and includes tutorials, examples, and API references. It's a great place to start and refer back to as you progress.
    
2. **PyTorch Tutorials**: These tutorials cover a wide range of topics from basic concepts to advanced topics such as distributed training and deploying models. They include:
    - **[Deep Learning with PyTorch: A 60 Minute Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)**: An in-depth, hands-on tutorial.
    
3. **[PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/starter/introduction.html)**: A high-level interface for PyTorch that helps to structure code and manage training loops. The tutorials here help you learn how to use Lightning to streamline the training process.


### Neural Networks

Neural networks are the backbone of LLMs, providing the structures and mechanisms necessary for these models to process and understand human language. By leveraging the strengths of neural networks in learning complex patterns, handling large-scale data, and adapting through transfer learning, LLMs achieve their impressive capabilities in language understanding and generation. Understanding neural networks is thus essential for grasping how LLMs function and excel in various language-related tasks.

You can get started with neural networks with some of the resources found below:
- [Neural Networks - 3Blue1Brown](https://www.youtube.com/watch?v=aircAruvnKk&ab_channel=3Blue1Brown)
- [Neural Networks - Josh Starmer](https://www.youtube.com/watch?v=CqOfi41LfDw&ab_channel=StatQuestwithJoshStarmer)
- [Visual Guide to NN - Jay Alammar](https://jalammar.github.io/visual-interactive-guide-basics-neural-networks/)

There is also a super cool tool to visualize what happens while training a neural network. You can check this out once you got the basics coverted:
- [NN Playground TF](https://playground.tensorflow.org/)

As an additional reading, do read up on Encoder-Decoder architectures in neural networks which is a concept which reverberates through most of LMs.

## LM Basics

A **large model (LM)** in the context of machine learning and artificial intelligence refers to a deep learning model that is characterized by a substantial number of parameters and is typically trained on large datasets. These models leverage the extensive computational power and large-scale data to achieve high performance on complex tasks, often surpassing smaller models in terms of accuracy and generalization.

LMs can be further divided into:
- Large Language Models (LLMs)
- Large Vision Models (LVMs)
- Large Audio Models (LAMs)

### LLMs
A Large Language Model (LLM) is a type of artificial intelligence model designed to understand, generate, and manipulate human language at a high level. These models are typically based on deep learning architectures, particularly transformers, and are trained on vast amounts of text data to learn patterns, structures, and semantics of the language.

In simpler terms, LLMs are super complex mathematical functions which are dumped with tremendous amounts of textual data and tasked with interpreting it, organizing it, transforming it and eventually generate new content on their own. These are massive models which required tons of infra to train, but once trained they can be reused / fine-tuned for multiple use cases.

> For example, let us look at the scale of GPT-3 which is a very large language model from OpenAI.
> - GPT-3 was trained on about 45 TB of data scraped from the web!
> - GPT-3 has about 175 Billion parameters & trained on 300 Billion tokens!
> - GPT-3 was trained on 1024 Nvidia V100 GPUs, costed $4.6M and 34 days to train!

Quite frankly LLMs have revolutionized the way we work not just in Software but in all fields. This has happened because of the wide scale of applications of a LLM:
- **Question-Answering system**
	- LLMs are used in customer support chatbots to provide instant answers to common customer queries, improving response times and customer satisfaction.
- **Sentiment Analysis**
	- Businesses use LLMs to analyze customer feedback from reviews and social media to gauge sentiment and improve their products
- **Text Summarization**
	- News agencies and academic researchers use LLMs to automatically summarize long articles or research papers, making it easier to digest large amounts of information quickly.
- **Machine Translation**
	- LLMs power services like Google Translate, enabling real-time translation of text and speech between multiple languages with high accuracy.

Some of the popular LLMs that you may have used/heard of:
- GPT-3, GPT-4o
- Mistral 7B
- Gemma

> A quick disclaimer on LLMs is that not all LLMs are generative in nature. You can read this post to understand differences between discriminative and generative models - [Generative Modelling](https://allenphilip93.github.io/posts/generative-deep-learning-ch-1/)
> ![Image](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*_8Bi4Yqz2Qnw4cE0aCEPaA.png)

### LVMs

A **Large Vision Model (LVM)** is a type of deep learning model designed specifically for tasks related to computer vision, which involves the interpretation and understanding of visual information from the world, such as images and videos. LVMs are built on advanced neural network architectures, often using Convolutional Neural Networks (CNNs) and more recently, Vision Transformers (ViTs).

Like the LLMs, there are tons of general purpose applications for the LVMs:
- **Image Classification**: Identifying objects or scenes within an image.
- **Object Detection**: Locating and identifying multiple objects within an image.
- **Image Segmentation**: Partitioning an image into segments for detailed analysis.
- **Image Generation**: Creating new images based on learned patterns.

Some of the popular LVMs that you may have used/heard of:
- Stability AI's - Stable Diffusion
- MidJourney
- Google's ImageGen

### LAMs

Large audio models, which are designed to process and understand audio data, are often referred to as **Large Audio Models (LAMs)**. These models leverage deep learning techniques to perform various tasks related to audio, such as speech recognition, music generation, sound classification, and more. Similar to LLMs and LVMs, LAMs are built on advanced neural network architectures and are trained on extensive datasets to achieve high performance.

Some of the popular applications for LAMs are as follows:
- **Automatic Speech Recognition (ASR)**: Converts spoken language into text.
- **Text-to-Speech (TTS)**: Converts written text to speech.
- **Speech Synthesis and Voice Cloning**: Generates human-like speech, often mimicking a specific person’s voice.
- **Music Generation**: Creates music compositions autonomously.

Some of the popular LAMs that you may have come across:
- Wave2Vec 2.0
- DeepSpeech
- Tacotron 2
- WaveNet

## Series Structure

There are tons of powerful LMs out there and the best part about it all is that many of the LMs are open-sourced! This means without paying $4.6M to build your own GPT-3 you can leverage the models out there and finetune them for your use case at no cost! It's an exciting time to be in and makes it all the more essential to join in on this bandwagon which is what this series attempts to do.

We will start our journey will LLMs and work our way to LVMs & LAMs since the core concepts are transferrable.

In simple terms, the breakdown is as follows:
- First, we attempt to understand how the textual data which is human understandable is passed to a ML model which only knows numbers
	- word2vec
	- Context vectors
	- Word Embeddings 

- Now that the input data is pre-processed and passed to the model, how does it learn? What are the challenges with conventional ML algorithms and how do LLMs do it differently?
	- Transformers
	- Attention Mechanism

- Next we want to categorize and understand the architecture of some of the most popular LLMs out there:
	- Categories of LLMs
	- BERT
	- GPT-2
	- GPT-3
	- LLaMA 2
	- LLaMA 3

- We know that training LLMs is expensive, so next is to look at how we can fine-tune existing open-source models
	- Transfer Learning
	- RAG
	- DPO
	- RLHF

This should largely conclude out journey on LLMs so next it will be LVMs!

- We start the same way, understanding how image are passed to LMs
	- Image Encodings
	- Image Embeddings

- Now that the input data is pre-processed and passed to the model, how does it learn? What are the challenges with conventional ML algorithms and how do LLMs do it differently?
	- VAE
	- GANs
	- Diffusion Models
	- Vision Transformer

- Interesting we can combine LLMs and LVMs to lead to interesting multi-modal use cases
	- Text2Image
		- Cross Attention
	- Image Edit
		- BG Removal
		- Style Change
		- Extend Frame
	- Text2Video

- Next we dive in to the internals and understand the architecture of some of the popular LVMs out there
	- Stable Diffusion
	- SoRA

- Fine-tuning opensource vision models for custom use cases
	- <>

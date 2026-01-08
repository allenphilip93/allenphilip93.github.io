---
layout: post
title: "GRPO: From Supervised Finetuning to Reinforcement Learning"
date: 2025-01-08
author: "Allen Philip J"
description: "Understanding the journey from pretraining to GRPO—covering SFT, LoRA, KL divergence, and how to evaluate RL training runs."
tags: [LLM, GRPO, Finetuning]
katex: true
---

ChatGPT generated a very apt intuitive analogy that's worth sharing:

> **Pretraining**: A student reads every book in the library (knows everything, but is opinionated and unfiltered).
>
> **SFT**: The teacher gives them model answers for exam-style questions (so they know how to behave in a test).
>
> **GRPO**: A coach gives live feedback as they practice mock exams ("more concise," "friendlier tone," "don't make stuff up")—shaping subtle preferences that are hard to encode in one-shot labels.

## Why Supervised Finetuning?

To understand Supervised Finetuning (SFT), it helps to take a step back and understand pretraining.

During pretraining, LLMs learn grammar, syntax, build world knowledge from a vast diverse corpus of text, and even develop some reasoning capabilities. Models like GPT take an input sentence and predict the next word as output. Though the model builds a strong understanding of the world, it's still not ready to be a helpful assistant—pretrained models focus on predicting the next word, not on following instructions or being helpful.

That's where SFT comes in and "finetunes" the model for specific tasks (in our case, to be a helpful instruction-following assistant). We train the model on curated instruction-response pairs.[^1]

[^1]: For example, conversation dumps from ChatGPT and Gemini can serve as general-purpose instruction finetuning datasets.

From my understanding, SFT works because the capacity to "learn" in a given LLM is finite. When we do pretraining, it's a general-purpose beast that is good at a lot of things. By performing SFT, we trade off learning in one part of the model to make certain aspects very strong. A pretrained model is a jack of all trades; SFT makes it a master of some.

## How to Finetune?

### Continued Pretraining

One way to get an LLM really good at a task is to make it see as much of that data as possible during pretraining. So why not just resume pretraining with tons of data targeting a specific task?

This makes sense if the task requires a fundamental shake-up of the LLM's world understanding. This is called **continued pretraining** or **domain adaptation**.

For example, if we take GPT-3 and want to use it for legal domain use cases, it may not have the domain vocabulary or semantics from its original pretraining. It makes sense to resume pretraining with a vast labeled legal dataset.

The trade-off: this is an **expensive and time-consuming process**.

### LoRA: Low-Rank Adaptation[^2]

[^2]: Hu, E., et al. "LoRA: Low-Rank Adaptation of Large Language Models." ICLR 2022.

But imagine I want to take my LLM and make it speak in a medieval tone. The LLM is already smart and has learned the grammar and semantics of English. This task requires the LLM to slightly adapt its understanding, not a ground-up rework.

This is where **LoRA** comes in and is incredibly effective. It modifies the LLM slightly, adjusting it towards performing the task we want better. This option is much faster and less compute-hungry, but the "learning" is limited.

Low-Rank Adapters (LoRA) is a technique where we freeze the LLM model weights and apply a sidecar trainable module to the layers we want to adapt.

We freeze the weights from the pretrained model and add a sidecar module/adapter which has only $4096 \times 2 \times r$ parameters (significantly fewer than $4096 \times 4096$ parameters).

**Key parameters:**

| Parameter | Purpose |
|-----------|---------|
| **Rank (r)** | Controls the amount of "learning" added to the model. More complex task → increase rank |
| **Alpha (α)** | Dampening effect to ensure LoRA activation doesn't overpower frozen model weights. Think of it as regularization. Not training well → increase alpha. Overfitting → decrease alpha |

### Which Layers to Apply LoRA?

Another key decision is which layers to apply LoRA adapters to:

- **Attention modules** capture semantic relationships between tokens. If you want SFT to change that, add LoRA adapters to `q_proj`, `k_proj`, `v_proj`, and `o_proj` (the MLPs that build QKV tensors)

- **MLP modules** (or experts) learn domain-specific information and house a lot of the "knowledge" in the LLM. If you want SFT to change the domain knowledge, add LoRA adapters to `up_proj` and `down_proj`

## Reinforcement Learning Basics

RL is a machine learning method where a model learns by taking decisions in an environment that maximize certain reward signals. Unlike SFT, this is an **unsupervised technique** and relies on reward signals to "learn".

Interestingly, RL techniques don't need us to explicitly give the answer—they expect the model to find it. This means training with RL is tricky. We can think of training as searching for a needle in a haystack:

- The goal is to find a small subset of model weights in a very large dimensional superset
- SFT makes it simpler by giving direct feedback of how far the current model weights are
- RL gives very little feedback—just tells if the model is doing good or bad

### How Does RL Work Then?

The crazy part is that even a **noisy signal** gives enough directional bias for the model to train. The moment the model gets feedback that it's getting more reward, the gradient update shifts the probability distribution towards it.

Still, there's a risk that the model might go around in circles since the signal is noisy. Solutions:

- **Batching**: Rewards are averaged across lots of samples → noise cancels out → stable gradient signal emerges. More the merrier!
- **KL penalty**: Ensures the new distribution doesn't drift too far from the SFT distribution

## KL Divergence

KL divergence is a measure of how different two probability distributions are. For LLMs:

- Distribution 1 = SFT model's token probabilities
- Distribution 2 = GRPO-updated model's token probabilities

By keeping an eye on KL divergence, we ensure the model doesn't drift too far during GRPO:

- RL techniques are exploratory and don't have fine-grained control like SFT
- KL penalty reminds the model if it has drifted too far in an attempt to maximize rewards
- For example, the model could find a way to game the reward

## Structuring Rewards

Given how important reward functions are to GRPO, it makes sense to understand how these functions are typically structured:

| Type | Description | Pros/Cons |
|------|-------------|-----------|
| **Binary (0/1)** | Simple yes/no signal | Direction often not clear enough for training |
| **Scalar (1-10)** | Model knows how well it's doing | Hard to consistently rate (is 6/10 vs 8/10 for politeness meaningful?) |
| **Pairwise (A > B)** | Humans judge "which is better?" | More natural for humans, but noisy and evens out only in aggregate |

## GRPO: Group Relative Policy Optimization[^3]

[^3]: Shao, Z., et al. "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models." 2024.

Every RL training has 3 key components:

- **Policy**: The model being trained
- **Action**: Model generation (output tokens)
- **Rewards**: Custom reward functions that nudge the model to our preferences

GRPO is an RL technique that uses **relative ranking of candidate responses** to generate a reward signal:

1. **Generation**: Model samples several candidate responses/actions
2. **Ranking**: Reward models rank them (e.g., give a scalar score)
3. **Policy Update**: This reward signal updates the policy (model weights)
   - KL loss ensures policy doesn't deviate too far
   - Entropy added so model doesn't get stuck on the same answer
   - Maximize reward by pushing probability mass toward higher-scoring responses

Essentially, GRPO tries to make the model more like the **better half of its answers**. Because of this, it relies on the fact that the model can already return good enough answers—hence we perform SFT first.

## Monitoring GRPO Training

GRPO tries to move the token distribution to one that maximizes rewards. Key metrics to monitor:

| Metric | What to Watch |
|--------|---------------|
| **KL Divergence** | Should increase from 0 as the model explores, then stabilize. If too high, the model has drifted too far from SFT distribution |
| **Reward** | Total reward should increase; standard deviation should decrease as responses become consistently good |
| **Entropy** | Measures exploration. Flat entropy early on may indicate reward gaming |
| **Grad Norm** | Large values mean the policy is updating aggressively |

**Common issues:**
- **Flat KL**: Model weights aren't updating—often due to uninformative rewards
- **KL not increasing enough**: Model isn't exploring sufficiently
- **Rewards plateau early**: May indicate reward hacking or insufficient signal

---

The key takeaway: GRPO is powerful but requires careful monitoring. The reward signal must be informative, the model must be capable of generating good responses (hence SFT first), and the training dynamics must show healthy exploration without drifting too far from the original distribution.

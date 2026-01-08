---
layout: post
title: "The Mathematics of Transformers"
date: 2024-10-15
author: "Allen Philip J"
description: "A mathematical deep dive into the transformer architecture, from attention mechanisms to positional encodings."
tags: [Transformers, Attention, ML]
katex: true
---

The transformer architecture has revolutionized machine learning since its introduction in 2017[^1]. In this post, we'll explore the mathematical foundations that make transformers so powerful.

## The Attention Mechanism

At the heart of every transformer lies the attention mechanism. Given queries $Q$, keys $K$, and values $V$, the scaled dot-product attention is computed as:

<figure>
  <img src="/assets/images/attention-diagram.svg" alt="Attention Mechanism Overview">
  <figcaption><strong>Figure 1.</strong> The attention mechanism computes a weighted sum of values based on query-key similarity.</figcaption>
</figure>

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

The scaling factor $\sqrt{d_k}$ prevents the dot products from growing too large, which would push the softmax into regions with extremely small gradients.

### Why Scale by $\sqrt{d_k}$?

Consider two vectors $q$ and $k$ with components drawn from $\mathcal{N}(0, 1)$. Their dot product has:

$$
\mathbb{E}[q \cdot k] = 0, \quad \text{Var}(q \cdot k) = d_k
$$

Without scaling, the variance grows with dimension, causing softmax saturation.

## Multi-Head Attention

<figure>
  <img src="/assets/images/transformer-architecture.svg" alt="Transformer Architecture">
  <figcaption><strong>Figure 2.</strong> Simplified transformer encoder block showing multi-head attention and feed-forward layers.</figcaption>
</figure>

Rather than performing a single attention function, transformers use **multi-head attention**[^2]:

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O
$$

where each head is computed as:

$$
\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$

> Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions.
>
> <cite>Vaswani et al., "Attention Is All You Need"</cite>

### Multi-Head Attention Parameters

| Model | Heads | d_model | d_k | Parameters |
|-------|-------|---------|-----|------------|
| BERT-Base | 12 | 768 | 64 | 110M |
| BERT-Large | 16 | 1024 | 64 | 340M |
| GPT-2 | 12 | 768 | 64 | 117M |
| GPT-3 | 96 | 12288 | 128 | 175B |
| LLaMA-7B | 32 | 4096 | 128 | 7B |
| LLaMA-70B | 64 | 8192 | 128 | 70B |

## Positional Encodings

Since transformers have no recurrence, we need to inject positional information. The original paper uses sinusoidal encodings:

$$
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

$$
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

This encoding has useful properties:

- **Bounded values**: Always in $[-1, 1]$
- **Unique positions**: Each position gets a distinct encoding
- **Relative positions**: $PE_{pos+k}$ can be represented as a linear function of $PE_{pos}$

## Implementation in PyTorch

Here's a clean implementation of scaled dot-product attention:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention mechanism."""

    def __init__(self, temperature: float, dropout: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Compute attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.temperature

        # Apply mask (for causal attention or padding)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))

        # Softmax and dropout
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Weighted sum of values
        output = torch.matmul(attn, v)

        return output, attn
```

And here's a complete multi-head attention implementation:

```python
class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Linear projections
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(
            temperature=math.sqrt(self.d_k),
            dropout=dropout
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        batch_size = q.size(0)

        # Project and reshape: (batch, seq, d_model) -> (batch, n_heads, seq, d_k)
        q = self.w_q(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # Apply attention
        output, attn_weights = self.attention(q, k, v, mask)

        # Concatenate heads and project
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.w_o(output)

        return output
```

## Complexity Analysis

The computational complexity of self-attention is:

| Operation | Time Complexity | Space Complexity |
|-----------|----------------|------------------|
| $QK^T$ computation | $O(n^2 \cdot d)$ | $O(n^2)$ |
| Softmax | $O(n^2)$ | $O(n^2)$ |
| Attention × V | $O(n^2 \cdot d)$ | $O(n \cdot d)$ |

This quadratic complexity in sequence length $n$ motivates research into efficient attention mechanisms[^3] like:

- **Linear attention**[^4] (Katharopoulos et al., 2020)
- **Sparse attention** (Child et al., 2019)
- **FlashAttention**[^5] (Dao et al., 2022)

## Key Takeaways

1. **Attention is a weighted aggregation** based on query-key similarity
2. **Scaling prevents gradient issues** in high dimensions
3. **Multi-head attention** enables diverse feature learning
4. **Positional encodings** inject sequence order information
5. **Quadratic complexity** remains a key challenge for long sequences

---

Understanding these mathematical foundations is crucial for optimizing transformers in production systems. In future posts, we'll explore how FlashAttention achieves IO-aware tiling to dramatically improve performance.

[^1]: Vaswani, A., et al. "Attention Is All You Need." NeurIPS 2017.
[^2]: Multi-head attention was a key innovation enabling parallel processing of different relationship types.
[^3]: The quadratic complexity becomes prohibitive for sequences longer than a few thousand tokens.
[^4]: Katharopoulos, A., et al. "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention." ICML 2020.
[^5]: Dao, T., et al. "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." NeurIPS 2022.

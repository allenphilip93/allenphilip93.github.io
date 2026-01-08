---
layout: post
title: "Flash, Fused and Fast Attention"
date: 2024-09-23
author: "Allen Philip J"
description: "A deep dive into FlashAttention and GPU optimization for transformer attention mechanisms."
tags: [Attention, GPU, Optimization]
katex: true
---

Around 2020, a paradigm shift occurred in attention research. Rather than developing mathematically clever but impractical alternatives like Sparse Attention, Linformer, and Performer, researchers pivoted to engineering-focused optimization through:

- CUDA-level kernel fusion combining multiple operations
- Avoiding materialization of large intermediate tensors
- Hardware-specific optimizations (tiling, pipelining, register management)
- Tensor core utilization

## Memory Hierarchy Overview

<figure>
  <img src="/assets/images/gpu-memory.svg" alt="GPU Memory Hierarchy">
  <figcaption><strong>Figure 1.</strong> GPU memory hierarchy showing HBM, SRAM, and registers with their respective capacities and bandwidths.</figcaption>
</figure>

Modern GPU architecture consists of:

- **HBM (High Bandwidth Memory):** 40-80GB on A100, ~1.5-2.0TB/s bandwidth
- **SRAM (Shared Memory):** ~192KB per streaming multiprocessor, ~19TB/s bandwidth

The key insight: compute speed vastly outpaces memory bandwidth, making data movement the bottleneck rather than computation itself.

## Flash Attention 1[^1]

[^1]: Dao, T., et al. "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." NeurIPS 2022.

### Core Problem

Standard transformer attention materializes a full $O(n^2)$ attention matrix, creating massive memory I/O overhead.

The standard attention computation is:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

### Solution

IO-aware tiling and recomputation strategies:

1. Break attention into blocks fitting in fast SRAM
2. Compute softmax on-the-fly without storing intermediate matrices
3. Recompute attention during backward pass using only saved statistics
4. Fuse operations into single CUDA kernels

**Memory Complexity:** Reduces from $O(n^2)$ materialized matrix to $O(n)$ for output storage.

### Performance Comparison

<table>
  <thead>
    <tr>
      <th>Method</th>
      <th class="center">Memory</th>
      <th class="center">Speed (A100)</th>
      <th class="center">Exact?</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><span class="indicator blue"></span>Standard Attention</td>
      <td class="center">O(n²)</td>
      <td class="center">1x (baseline)</td>
      <td class="center highlight">Yes</td>
    </tr>
    <tr>
      <td><span class="indicator light-blue"></span>Sparse Attention</td>
      <td class="center">O(n√n)</td>
      <td class="center"><u>1.5-2x</u></td>
      <td class="center">No</td>
    </tr>
    <tr>
      <td><span class="indicator light-blue"></span>Linear Attention</td>
      <td class="center">O(n)</td>
      <td class="center"><u>2-3x</u></td>
      <td class="center">No</td>
    </tr>
    <tr>
      <td><span class="indicator green"></span>FlashAttention</td>
      <td class="center highlight">O(n)</td>
      <td class="center highlight"><u>2-4x</u></td>
      <td class="center highlight">Yes</td>
    </tr>
  </tbody>
</table>

## Flash Attention 2[^2]

[^2]: Dao, T. "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning." 2023.

FlashAttention 1 achieved only 30-50% GPU utilization. FA2 improvements:

- **Reduced non-matmul operations:** Store logsumexp instead of separate max/sum statistics
- **Improved parallelism:** Parallelize over sequence blocks, not just batch/heads
- **Warp optimization:** Split queries across warps instead of keys/values to reduce synchronization overhead

Result: 2-4x wall-clock speedup, up to 73% peak performance utilization on A100.

### FlashAttention Version Comparison

| Feature | FA1 | FA2 | FA3 |
|---------|-----|-----|-----|
| GPU Utilization | 30-50% | ~73% | ~85% |
| Parallelization | Batch/Heads | + Sequence | + Warp-specialized |
| FP8 Support | No | No | Yes |
| Target Hardware | Ampere | Ampere | Hopper |
| Relative Speedup | 1x | 2x | 2.5-3x |

## Flash Attention 3[^3]

[^3]: Shah, J., et al. "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision." 2024.

Hardware-specific optimizations for Hopper GPUs (H100/H200):

### Warp-Specialization

Producer warps load data while consumer warps compute, using async barriers and circular buffers to hide latency.

### GEMM-Softmax Pipelining

Overlaps matrix multiplication and softmax computation across warp groups, utilizing both tensor cores and special function units simultaneously.

### FP8 Support

Block-wise quantization (per-tile) with incoherent processing spreads outliers, nearly doubling throughput via native FP8 cores.

## Code Examples

### FlashAttention-2 Usage

```python
from flash_attn import flash_attn_func

# Q, K, V shape: [batch, seqlen, nheads, headdim]
# Must be float16/bfloat16
out = flash_attn_func(q, k, v, dropout_p=0.0,
                      causal=False, softmax_scale=None)
```

### FlashAttention-3 Installation

```bash
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
git checkout flashattention-3
pip install -e .
```

### Benchmark Results (H100)

| Sequence Length | Standard (ms) | FA3 (ms) | Speedup |
|-----------------|---------------|----------|---------|
| 2,048 | 4.2 | 1.1 | 3.8x |
| 4,096 | 16.8 | 3.2 | 5.3x |
| 8,192 | 67.1 | 10.4 | 6.5x |
| 16,384 | 268.5 | 35.7 | 7.5x |

## Conclusion

Attention is everywhere in modern AI systems. Optimizing it enables faster training, cheaper inference, and energy efficiency. Future developments include wider FP8 adoption, alternative kernel implementations (cuDNN, Triton), and potential hardware solutions moving attention out of the critical path.

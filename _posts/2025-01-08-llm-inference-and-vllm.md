---
layout: post
title: "LLM Inference: From KV Caching to vLLM"
date: 2025-01-08
author: "Allen Philip J"
description: "A comprehensive guide to LLM inference optimization, covering KV caching, PagedAttention, continuous batching, and vLLM configuration."
tags: [LLM, Inference, Optimization]
katex: true
---

Large Language Models are largely decoder-only and autoregressive in nature. They generate tokens one by one, building and shifting the context window with each step. KV caching allows us to reuse the KV tensors for tokens as we shift the context window.

Building on this, we can breakdown LLM inference into two distinct phases:

- **Prefill**: Calculate and store the KV cache for the input tokens (first forward pass)
- **Decoding**: Reuse the cache and compute KV cache for the previous output token

## Key Metrics

Understanding LLM inference requires familiarity with several important metrics:

| Metric | Description |
|--------|-------------|
| **Time To First Token (TTFT)** | Latency for first forward pass (prefilling stage) |
| **Inter-Token Latency (ITL)** | Average latency to generate each token in decoding |
| **Model Size** | Memory required to store model weights |
| **Peak Memory** | Maximum memory usage during generation |
| **KV Cache Size** | Memory required to store the KV cache |
| **Token Throughput** | Number of tokens generated per second |
| **Request Throughput** | Number of requests completed per second |

## Things to Keep in Mind

- As models become larger with more parameters, so does the model size
- In the prefilling stage, self-attention exhibits **quadratic computational complexity** with sequence length—larger context windows mean longer first token latency
- KV cache is stored in HBM/GDDR which needs to be moved to on-chip storage during lookup—an expensive IO operation (remember the FlashAttention paper)[^1]

[^1]: This memory movement is often the bottleneck in LLM inference, not compute.

## KV Caching

Having more GPU memory can help improve overall throughput for several reasons:

- Larger KV caches allow more tokens to be cached, enabling faster lookups and higher throughput
- Larger batch sizes for inference improve GPU utilization

But this is only helpful if certain criteria are met:

- **High memory bandwidth** — Otherwise it's still a bottleneck
- **Long input/output contexts** — For one-shot prompts with minimal autoregressive decoding, caching doesn't add much value
- **Compute is not the bottleneck** — With 128K full attention, KV cache becomes compute-bound
- **Parallel sampling or beam search** — The benefit of KV caching the prompt is multiplied!

## PagedAttention and vLLM

Existing systems struggle because the KV cache memory for each request is huge and grows/shrinks dynamically. When managed inefficiently, this memory can be significantly wasted by fragmentation and redundant duplication, limiting batch size.[^2]

[^2]: [vLLM Paper](https://arxiv.org/pdf/2309.06180) — Current data indicates that handling an LLM request can be 10x more costly than a standard keyword query.

vLLM promises several key advantages:

- Near-zero waste in KV cache memory utilization
- Flexible sharing of KV cache within and across requests

### The Problem with Existing Systems

Existing systems suffer from two main problems:

1. **Pre-allocation waste**: To store the KV cache, they pre-allocate a contiguous chunk of memory with the request's maximum length. This causes both internal and external fragmentation—small requests underutilize memory, and new requests struggle to find contiguous blocks.

2. **No memory sharing**: KV cache of sequences are stored in separate contiguous spaces. In cases like parallel sampling (generating multiple responses for the same prompt) or beam search, they don't share KV cache and create redundant memory blocks.

### How PagedAttention Works

PagedAttention is inspired by how operating systems handle memory fragmentation and sharing:

- Divides the KV cache into **blocks**, each containing attention keys and values for a fixed number of tokens
- Blocks are **not necessarily contiguous** in memory
- Blocks are **allocated on demand**
- Same-size blocks address the external fragmentation problem
- Enables **block-level sharing** of KV caches for same/different requests

PagedAttention works purely due to optimized kernels—the block lookups are minimal and can be overlooked. Though it's called PagedAttention, it operates at a layer below the actual attention computation. PagedAttention fetches KV values efficiently; the attention itself can use something like FlashAttention.[^3]

[^3]: Keep in mind that FlashAttention works best with large sequence lengths, so varied sequence lengths might not benefit as much.

## Continuous Batching

Batching LLM requests is non-trivial for two reasons:

1. **Requests arrive at different times** — A naive strategy would make requests wait for late ones or delay until earlier ones complete
2. **Vastly different sequence lengths** — Padding to the largest input/output sequence length wastes GPU memory and compute

vLLM introduces **Cellular Batching** and **iteration-level scheduling**:

- After each iteration (forward pass), completed requests are removed and new ones added
- A new request can be processed after waiting for just **one iteration**, not entire requests
- Special GPU kernels eliminate the need to pad inputs and outputs

This drives up GPU utilization, but there's little guarantee on batch composition—you could have prefill-only batches, decode-only batches, or mixed. Due to non-uniform units of compute, GPU utilization and latencies can be unpredictable.

## Chunked Prefill

As of vLLM v1, chunked prefill is enabled by default. When enabled, the scheduler **prioritizes decode over prefill**.

**Example**: Imagine `max_num_batched_tokens` is 1000. There are 50 ongoing conversations (decode requests) and 2 new requests in queue:
- Request A (FCFS): prompt length 2000
- Request B: prompt length 300

Here's how the scheduler builds the next batch:

1. **Prioritize Decodes**: Add 1 token for each of the 50 decode requests → 50 tokens used, 950 remaining
2. **Service Prefills (FCFS)**: Request A is first. Can 2000 tokens fit in 950? No.
3. **Chunk it**: Take the first 950 tokens from Request A's prompt

**Final Batch**: 50 decode tokens + 950 prefill tokens = 1000 tokens. Request B remains in queue for the next iteration.

### Sarathi: Analysing Prefill & Decode

The Sarathi paper provides deeper analysis of prefill and decode phases.[^4]

[^4]: [Sarathi Paper](http://arxiv.org/pdf/2308.16369)

Key observations for LLaMa-13B with fixed sequence length of 1024:

- Prefill has almost **constant per-token cost**
- Decode varies significantly since it's **memory-bound**, not compute-heavy
- Majority of latencies go into matmuls in attention & MLP layers (others < 5% of total)

The difference in throughput scaling stems from the fact that the prefill phase computes **batched matrix-matrix multiplications** as opposed to the **vector-matrix multiplications** of the decode phase.

### How Chunked Prefill Works

Mathematically, it makes a simple claim: if $Y = \text{model}(X)$ captures model inference, then:

$$Y_t = \text{model}(X[:t]) = Y[:t]$$

This holds because LLM inference is **causal** in nature. More precisely, chunked prefill says:

$$Y_{ab} = \text{model}(X[a:b]) = Y[a:b]$$

This holds as long as we incrementally process chunks and cache prior KV tensors. Only attention computation requires all previous values up to the current token—the linear matmuls don't need context beyond what the input provides.

**Drawbacks to be cautious of:**

- By splicing large sequences into smaller ones, we decrease arithmetic intensity—finding the right chunk size for the hardware is important
- Chunked prefill reads from KV caches for computing incremental values, adding overhead

### Decode-Maximal Batching

Now that we can effectively chunk prefills, the goal is to combine them efficiently to maximize GPU utilization.

The simplest approach is to **prioritize decode** and ensure a ratio like 10:1 (decode:prefill) to saturate the GPU while decode processes quickly. This ratio is hardware and use-case specific.

Identifying a suitable chunk size involves a trade-off: smaller chunks piggyback more decodes but at the expense of lower prefill efficiency, whereas larger chunks are prefill-efficient but piggyback fewer decodes.

## PD Disaggregation (Experimental)

vLLM supports prefill-decode disaggregation, structured for extensibility since the connector implementation is environment-specific (NVLink, Infiniband, etc.).[^5]

[^5]: [vLLM PD Disaggregation Docs](https://docs.vllm.ai/en/stable/features/disagg_prefill.html)

To transfer KV cache:
- Lookup buffers exist at prefill and decode instances
- The prefill instance loads into the buffer while decode reads from it
- A pipe facilitates the transfer (implementation-specific)
- Once KV cache reaches decode, it can send a `drop_select` command to clear the prefill buffer

## vLLM Configuration Parameters

### GPU Memory Utilization

The fraction of total GPU memory that vLLM will pre-allocate for KV cache. Ranges from 0 to 1 (default is often 0.9).

Larger KV cache generally means higher throughput, especially for long sequences. In LLM inference, the decoding phase is often memory-bound.

### Max Model Length

The maximum context length (input + output tokens) the model can handle. If unspecified, it's derived from model configuration.

vLLM uses this to reserve enough KV cache space. A larger-than-required value means over-allocation and reduced concurrent requests.

### Max Num Batched Tokens

Maximum total tokens (across all sequences) processed in a single GPU iteration. Directly impacts batch size.

- **To reduce prefill impact**: LOWER this value (e.g., 2048-4096). Smaller budget means smaller prefill chunks, better inter-token latency.
- **To increase prefill throughput**: RAISE this value (e.g., 8192+). Larger budget allows bigger prefill chunks, improving time-to-first-token.

### Max Concurrent Sequences

Maximum number of concurrent requests. Directly limits in-flight requests—if GPU memory allows more but this is lower, it caps throughput.

### Enable Prefix Caching

**Default: True**. Caches KV states of common prompt prefixes across requests.

Significant for workloads with shared beginnings (multi-turn conversations, templated prompts, chatbots). Drastically reduces redundant computation and KV cache allocation.

### Quantization

vLLM supports dynamic quantization of BF16/FP16 models to FP8 without calibration:

```python
# Command line
--quantization="fp8"

# Or in constructor
LLM(..., quantization="fp8")
```

Uses "llmcompressor"—similar to torchao and modelopt. Static quantization with calibration data is also possible.

### Tensor Parallelism

Required when models don't fit on a single GPU. With fast GPU communication (NVLink), this approach makes sense.[^6]

[^6]: [vLLM Distributed Serving Docs](https://docs.vllm.ai/en/latest/serving/distributed_serving.html)

### Expert Parallelism

Currently EP and TP are exclusive in vLLM. EP design is largely based on DeepSeek-v2 with some generalization.

Current MoE mapping is sequential—there's no way to guarantee load balancing if certain experts are used more often.

### Compilation Config

By default, Inductor is used for general shape compilation. Inductor also allows compilation for specific shapes:

```bash
vllm serve meta-llama/Llama-3.2-1B \
  --compilation_config '{"compile_sizes": [1, 2, 4, 8]}'
```

This generates tailored kernels for batch sizes 1, 2, 4, and 8. When all shapes are static, auto-tuning is enabled for maximum performance.

For attention backends that are cudagraph-compatible, include attention in the cudagraph for improved decode speed (especially for smaller models):

```bash
--compilation-config '{"full_cuda_graph": true}'
```

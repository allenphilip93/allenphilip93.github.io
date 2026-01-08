---
layout: post
title: "GPU vs CPU: Matmul, Sine Waves, and the Myth of Speed"
date: 2024-06-14
author: "Allen Philip J"
description: "Understanding GPU performance—when parallel processing outweighs overhead, and when it doesn't."
tags: [GPU, Performance, PyTorch]
katex: true
---

This post presents a detailed performance analysis comparing CPU and GPU execution for common machine learning operations. Using PyTorch on an NVIDIA A100 GPU, we'll explore when GPUs provide actual performance benefits and when they might not be the optimal choice.

## The matmul Operation

Matrix multiplication is one of the most common operations when it comes to training ML models or using them for inference. So let's check out how a simple matmul operation holds up when we run it on CPU vs GPU.

This is the function we are going to benchmark:

```python
@timeit(repeats=5)
def matmul(x: torch.Tensor, y: torch.Tensor):
    return torch.matmul(x, y)
```

There are tons of ways to benchmark functions in Python but in this case we're using a custom `timeit` decorator which executes the function `n` times and returns the average.

Let's start getting some numbers for CPU:

```python
times_cpu = []

for i in range(16):
    print(f"Iteration: #{i}")
    dim = 2 ** i
    x = torch.rand(dim, dim)
    y = torch.rand(dim, dim)

    times_cpu.append(matmul(x, y))
```

If we plot the CPU times, we see an exponential increase in time as we increase the dimension of the input matrix—which makes sense.

## matmul on GPU

Now let's tweak our function to run on GPU. Essentially, we're loading the tensors to the GPU, performing `matmul`, and returning to CPU.

```python
@timeit(repeats=5)
def matmul(x: torch.Tensor, y: torch.Tensor):
    return torch.matmul(x.cuda(), y.cuda()).cpu()

times_gpu = []
for i in range(16):
    print(f"Iteration: #{i}")
    dim = 2 ** i
    x = torch.rand(dim, dim)
    y = torch.rand(dim, dim)

    times_gpu.append(matmul(x, y))
```

When we compare the time taken from CPU & GPU for `matmul` we can make a couple of observations:
- GPU starts beating CPU only at much larger dimensions where the multiprocessing capabilities really come into play
- There is an odd bump at the start of the GPU curve—this is due to GPU kernel initialization which is required the first time but not afterwards

With all the modern CPU processors, the benefit of GPUs really kicks in at larger scale only. So while optimizing, if your scale is not that large, maybe try running it on a CPU instead.[^1]

[^1]: This is a counterintuitive but important insight for ML practitioners.

## matmul on GPU without IO

In our benchmark function, we are performing a lot of IO moving the tensors back and forth between CPU & GPU. Let's try to evaluate the raw compute performance of the GPU instead:

```python
@timeit(repeats=5)
def matmul(x: torch.Tensor, y: torch.Tensor):
    return torch.matmul(x, y)

times_gpu_noio = []
for i in range(16):
    print(f"Iteration: #{i}")
    dim = 2 ** i
    x = torch.rand(dim, dim).cuda()
    y = torch.rand(dim, dim).cuda()

    times_gpu_noio.append(matmul(x, y))
```

When we plot and compare the times for the 3 runs, it looks like the GPU compute is almost constant! Does this mean all the work for the GPU run is going towards IO?

Answer is no! We have to remember that GPU is inherently async in nature and simple timer statements are not adequate to capture the actual time taken for the GPU compute. In our example we basically take a difference of the `time.time()` value before and after the function executes, but the GPU may still be running operations asynchronously.

To fix this, we need to synchronize:

```python
@timeit(repeats=5)
def matmul(x: torch.Tensor, y: torch.Tensor):
    z = torch.matmul(x, y)
    torch.cuda.synchronize()
    return z
```

If you are profiling GPU components, ensure you synchronize to get the actual time taken. Otherwise you're just calculating the time it takes to submit the kernel to GPU![^2]

[^2]: This is one of the most common profiling mistakes in GPU programming.

Now let's rerun the experiment and the world makes sense again! If you look at the graph you'll realize the actual compute is just 50%, the other 50% is the IO. For optimization this is very important—you'll know if you need to make the algorithm faster or if you're memory constrained.

## Nested sine Operation

Let's take a detour and explore a slightly different function—a nested `sine`:

```python
def nested_sin(x: torch.Tensor):
    x = torch.sin(x)
    x = torch.sin(x)
    x = torch.sin(x)
    x = torch.sin(x)
    return x

def nested_sin_gpu(x: torch.Tensor):
    x = x.to('cuda')
    x = torch.sin(x)
    x = torch.sin(x)
    x = torch.sin(x)
    x = torch.sin(x)
    torch.cuda.synchronize()
    y = x.cpu()
    return y
```

To ensure accurate profiling and correct synchronization, you should synchronize the GPU before moving the tensor back to the CPU. This guarantees that all GPU operations are completed before you start transferring data back.[^3]

[^3]: Synchronizing after `x.cpu()` is redundant because `x.cpu()` will wait for all preceding operations on the GPU to complete before it starts transferring data.

Unlike `matmul`, `sine` is not very computationally intensive. Let's run the same experiment to see how this stacks up.

This is quite the surprise! CPU is faster than GPU at scale (for tensor of size $2^{32}$) but interestingly the GPU time without IO is extremely small. We can make two conclusions:
- Our `nested_sin` is a pretty simple function
- GPU can overcomplicate simple functions

GPUs are specialized compute great at massive parallelizable tasks, but the value add drops when the task is simple where the overheads of GPU outweigh the benefits.[^4]

[^4]: This is why not everything should be moved to GPU—the overhead can dominate for simple operations.

## Profiling nested sine IO

Let's take a deeper look to understand where the overhead is coming from. Based on the data at hand, we suspect IO to be the culprit. So let's get some evidence using `torch.profiler`.

The torch trace confirms our suspicions:
- The CPU compute for the nested `sine` is super quick
- The GPU compute shows a lot of `aten::to_copy` and `memcpy`—these are the IO operations to move the input tensor to GPU and back
- The actual kernel for `sine` is super small, which reinforces our statement that it's a simple operation that doesn't get a lot of optimization from GPU

Operations like `matmul` are super common in ML so most GPUs are explicitly optimized to handle it far more efficiently (check out tensor cores).[^5]

[^5]: Tensor cores are specialized hardware units in NVIDIA GPUs designed specifically for matrix operations.

## Warmup for nested sine

Let's look at the trace of our `nested_sin_gpu` for a smaller input tensor ($2^{10}$ size) and run the function twice:

- The first observation is the long `aten::to` slice on the CPU thread for the first run. This is because PyTorch initializes the CUDA context the first time—this involves setting up various resources and only happens once per process for each GPU.

- Next is the `aten::sin` kernel which is long for the first time but far shorter afterwards. The first time a particular CUDA kernel is used, it needs to be loaded into GPU memory. Subsequent calls use the already-loaded kernel. PyTorch and CUDA also employ JIT compilation for certain operations.

Always warmup your environment before profiling or deployment to ensure more consistent performance.[^6]

[^6]: A warmup phase of 2-3 iterations is typically sufficient for most workloads.

## Overlap CUDA Streams

In all our profiling so far, notice how all the GPU execution happens on the same stream. Let's see if overlapping the following tasks on separate streams improves performance:
- Upload input tensor to CUDA
- Compute the nested `sine`
- Download output tensor from CUDA

```python
from torch.cuda import Stream

upload_stream = Stream()
compute_stream = Stream()
download_stream = Stream()

def nested_sin_gpu_streams(x: torch.Tensor):
    x_gpu = torch.empty_like(x, device='cuda')

    # Stream 1: Upload data to GPU
    with torch.cuda.stream(upload_stream):
        x_gpu.copy_(x, non_blocking=True)
    upload_stream.synchronize()

    # Stream 2: Perform computation
    with torch.cuda.stream(compute_stream):
        y_gpu = torch.sin(x_gpu)
        y_gpu = torch.sin(y_gpu)
        y_gpu = torch.sin(y_gpu)
        y_gpu = torch.sin(y_gpu)
    compute_stream.synchronize()

    # Stream 3: Download data to CPU
    with torch.cuda.stream(download_stream):
        y = y_gpu.cpu()
    download_stream.synchronize()

    return y
```

Warmup applies to streams as well. Each time you create a stream with `torch.cuda.Stream` it needs to be initialized.

In this case we don't see much of a benefit, but it can help in cases which are strongly IO heavy—though the compiler itself can often make such optimizations on the fly.

## Compiled Kernels with torch.jit.script

Let's bring our attention to the 4 kernel calls that happen back and forth from CPU & GPU for `aten::sin`. The GPU is loading and executing the same kernel every time just with different values. What if we can combine the kernels to execute just once on the GPU?

That's exactly what `torch.jit.script` does:

```python
def nested_sin(x: torch.Tensor):
    x = torch.sin(x)
    x = torch.sin(x)
    x = torch.sin(x)
    x = torch.sin(x)
    return x

compiled_nested_sin = torch.jit.script(nested_sin)
```

Warmup helps here as well—the first time you run the compiled function, it'll execute as-is and compile it. You'll see the 4 kernel calls for the first run. But after that the function is compiled, and you'll see only the fused kernel.

Compiling kernels can be a very powerful tool—in this example we reduce the `nested_sin` execution time by almost 50% from 100us to 50us approx. But the problem is still memory bound so CPU will still be faster here!

---
title: TensorRT - From Frustration to Production
date: 2025-04-11 10:15:00 +0530
categories: [Optimization]
tags: [ML, GPU]
math: false
pin: false
image:
  path: https://wallpapers.com/images/hd/manufacturing-2048-x-1120-wallpaper-dty69xwzaexx6kn7.jpg
  alt: TensorRT - From Frustration to Production
---

## TensorRT in Practice

TensorRT promises significant performance improvements for deep learning inference. Though it varies from case to case, I have consistently seen minimum reductions in latency of about 15% on A100s/H100s to even 80% in some cases. And I'm not even talking about quantization so far.

But the journey from development to production is often fraught with unexpected challenges. This guide aims to bridge the gap between theory and practice, helping you navigate common pitfalls and achieve production-ready deployments.

## Installing the Right Version

Version compatibility is crucial for TensorRT success. Here's a systematic approach:

1. **Docker Image Compatibility**
   Ensure that the cuda version in your base match is support by the TensorRT version you wish to use.
   ```bash
   FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
   ```

2. **Dependency Matrix**
   | Component | Version Check |
   |-----------|---------------|
   | CUDA | `nvidia-smi` |
   | Python | `python --version` |
   | TensorRT | `python3 -c "import tensorrt; print(tensorrt.__version__)"` |
   | GPU Driver | `nvidia-smi --query-gpu=driver_version --format=csv,noheader` |

3. **Installation Script**
   ```bash
    #!/bin/bash

    # Remove existing installations
    sudo apt-get remove --purge 'tensorrt*' 'libnvinfer*' 'python3-libnvinfer*'
    pip uninstall tensorrt

    # Install specific version via apt-get
    sudo apt-get install \
        tensorrt=10.4.0.26-1+cuda12.6 \
        libnvinfer10=10.4.0.26-1+cuda12.6 \
        libnvinfer-plugin10=10.4.0.26-1+cuda12.6 \
        libnvinfer-vc-plugin10=10.4.0.26-1+cuda12.6 \
        libnvinfer-lean10=10.4.0.26-1+cuda12.6 \
        libnvinfer-dispatch10=10.4.0.26-1+cuda12.6 \
        libnvonnxparsers10=10.4.0.26-1+cuda12.6 \
        libnvinfer-bin=10.4.0.26-1+cuda12.6 \
        libnvinfer-dev=10.4.0.26-1+cuda12.6 \
        libnvinfer-lean-dev=10.4.0.26-1+cuda12.6 \
        libnvinfer-plugin-dev=10.4.0.26-1+cuda12.6 \
        libnvinfer-vc-plugin-dev=10.4.0.26-1+cuda12.6 \
        libnvinfer-dispatch-dev=10.4.0.26-1+cuda12.6 \
        libnvonnxparsers-dev=10.4.0.26-1+cuda12.6 \
        libnvinfer-samples=10.4.0.26-1+cuda12.6 \
        python3-libnvinfer-dev=10.4.0.26-1+cuda12.6 \
        python3-libnvinfer=10.4.0.26-1+cuda12.6 \
        python3-libnvinfer-lean=10.4.0.26-1+cuda12.6 \
        python3-libnvinfer-dispatch=10.4.0.26-1+cuda12.6 \
        libnvinfer-headers-dev=10.4.0.26-1+cuda12.6 \
        libnvinfer-headers-dev=10.4.0.26-1+cuda12.6 \
        libnvinfer-headers-dev=10.4.0.26-1+cuda12.6 \
        libnvinfer-headers-plugin-dev=10.4.0.26-1+cuda12.6 \
        libnvinfer-headers-plugin-dev=10.4.0.26-1+cuda12.6 \

    pip install tensorrt==10.4.0

    dpkg -l | grep nvinfer

    # Dependencies for inference
    pip install cuda-python
    pip install cupy-cuda12x

    # Dependencies for fp8 quantization with TensorRT
    pip install git+https://github.com/NVIDIA/TensorRT.git#subdirectory=tools/onnx-graphsurgeon --no-build-isolation
    pip install polygraphy
    pip install nvidia-modelopt
    pip install pulp
    pip install torchprofile
   ```

## TRT Runtime for Inference

Using TensorRT for inference involves several key steps:

```python
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

class TRTInference:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

    def infer(self, input_tensor):
        # Allocate device memory
        d_input = cuda.mem_alloc(input_tensor.nbytes)
        d_output = cuda.mem_alloc(self.engine.get_binding_size(1))
        
        # Transfer data to GPU
        cuda.memcpy_htod_async(d_input, input_tensor, self.stream)
        
        # Execute inference
        self.context.execute_async_v2(
            bindings=[int(d_input), int(d_output)],
            stream_handle=self.stream.handle
        )
        
        # Transfer results back
        output = np.empty(self.engine.get_binding_shape(1), dtype=np.float32)
        cuda.memcpy_dtoh_async(output, d_output, self.stream)
        self.stream.synchronize()
        
        return output
```

> Always check out the Nvidia [Support Matrix](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html) before getting started.
{: .prompt-tip}

## Dealing with Attention for ONNX Export

Flash Attention and similar optimizations often cause ONNX export issues:

1. **Common Export Failures**
   - Flash Attention 2/3 operations not supported
   - XFormers custom kernels not mappable
   - Dynamic sequence lengths causing shape inference issues

2. **Solution: Using SDPA**
   ```python
   # Replace Flash Attention with SDPA
   from torch.nn.functional import scaled_dot_product_attention as sdpa
   
   class Attention(nn.Module):
       def forward(self, q, k, v):
           return sdpa(q, k, v, is_causal=True)
   ```

   > Keeping in mind the tensor shape for torch's sdpa is a bit different $(batch, nheads, seqlen, headdim)$ as compared to FA2/FA3 $(batch, seqlen, nheads, headdim)$
   {: .prompt-info}

3. **Export with Dynamic Shapes**
   ```python
   torch.onnx.export(
       model,
       (q, k, v),
       "attention.onnx",
       input_names=["query", "key", "value"],
       dynamic_axes={
           "query": {0: "batch", 1: "seq_len"},
           "key": {0: "batch", 1: "seq_len"},
           "value": {0: "batch", 1: "seq_len"}
       }
   )
   ```

## Common Pitfalls in TRT Engine Building

TensorRT's static compilation nature presents unique challenges when dealing with dynamic control flow in your model. Let's explore common scenarios and their solutions:

### Conditional Branching

Consider a forward pass with dynamic conditions:
```python
def forward(self, x):
    seqlen = x.size(1)
    if seqlen > 5000:
        return self.long_sequence_path(x)
    else:
        return self.short_sequence_path(x)
```

**The Problem:**
- TensorRT is a static compilation framework
- Dynamic branching prevents optimal graph optimization
- ONNX export only captures the path taken by the sample input

**What Actually Happens:**
1. The export succeeds but only traces one branch
2. The engine runs but may not handle all cases correctly
3. Performance optimizations are limited due to uncertainty

> **Note**: While newer ONNX opsets support conditional operations, TensorRT's static nature still imposes this limitation.
{: .prompt-warning}

### State Management

TensorRT engines are inherently stateless, which affects two common scenarios:

1. **Cached Values**
   ```python
   class Model(nn.Module):
       def __init__(self):
           self.cache = {}  # Problematic for TensorRT
           
       def forward(self, x):
           if x not in self.cache:
               self.cache[x] = self.compute(x)
           return self.cache[x]
   ```

   **Solution:**
   - Remove caching logic before export
   - Consider alternative optimization strategies
   - Use TensorRT's built-in memory optimization

2. **Scalar Inputs and Intermediate Values**
   ```python
   def forward(self, x, temperature=1.0):  # Scalar parameter
       return x * temperature
   ```

   **Challenges:**
   - Scalar values are often hardcoded during compilation
   - Only tensor inputs are supported in the final engine
   - Dynamic parameter adjustment becomes impossible

   **Workarounds:**
   1. Use fixed values that represent your typical use case
   2. Compile specific submodules instead of the full model
   3. Create separate engines for different parameter values

> **Tip**: When dealing with complex models, consider compiling only the computationally intensive parts (e.g., transformer blocks) rather than the entire model. This approach often provides the best balance between flexibility and performance.
{: .prompt-tip}

### Best Practices

1. **Design for Static Execution**
   - Avoid dynamic control flow in critical paths
   - Use fixed shapes where possible
   - Consider separate engines for different scenarios

2. **Modular Compilation**
   - Break down complex models into stateless components
   - Compile performance-critical sections independently
   - Maintain flexibility in the outer control flow

3. **Validation Strategy**
   - Test with various input sizes and shapes
   - Verify behavior matches the original model
   - Profile performance across different scenarios

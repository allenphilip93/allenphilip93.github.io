---
layout: post
title: "Understanding Inference Optimization Frameworks"
date: 2024-08-04
author: "Allen Philip J"
description: "A comprehensive guide to modern deep learning inference optimization, from compilers to runtimes."
tags: [Optimization, TensorRT, PyTorch]
katex: true
---

Modern deep learning models require efficient execution to meet production demands. While the core logic of our code defines what we want to achieve, its execution depends on numerous lower-level components that translate our intentions into machine instructions. This translation layer, known as the runtime, is crucial for model inference performance.

Runtime optimization in deep learning focuses on improving the efficiency of model inference execution. As models grow larger and GPU resources remain expensive, optimizing runtime performance becomes essential for both cost savings and better user experience.

The default PyTorch runtime, while suitable for development and small-scale inference, includes significant overhead (as demonstrated by projects like [llm.c](https://github.com/karpathy/llm.c)). This overhead is inherent in its general-purpose design. For production-scale deployment, specialized optimization frameworks become necessary.

Modern inference optimization frameworks consist of two key components:
- **Compiler**: Transforms high-level model code into optimized low-level representations
- **Runtime**: Manages the execution of compiled code with minimal overhead

## Compiler Architecture

The compiler transforms high-level PyTorch model code into optimized low-level representations through a multi-stage process.

### Graph Acquisition

Inference can be viewed as executing a sequence of operations to transform input data into predictions. While inputs may vary, the operation sequence remains constant (and often, so do the input shapes).

Graph acquisition converts this operation sequence into a computational graph where:
- Nodes represent individual operations
- Edges represent data flow and execution dependencies

Popular tools for graph acquisition include:
- **Torch FX**: PyTorch's native graph capture tool
- **Torch Dynamo**: Dynamic graph tracing for PyTorch
- **ONNX**: Cross-platform graph representation format

### Graph Lowering

Graph lowering transforms the high-level computational graph into a lower-level intermediate representation (IR) that better matches the target hardware's execution model. This transformation enables:

- Direct mapping of PyTorch operations to hardware-specific kernels
- Removal of inference-irrelevant components (e.g., dropout, gradient computation)
- Static optimization of deterministic control flow
- Better optimization opportunities through hardware-aware transformations

Common lowering implementations include:
- **AOTAutograd**: Ahead-of-time autograd transformation
- **FX passes**: PyTorch's graph transformation passes
- **ONNX Runtime**: Cross-platform optimization passes
- **TensorRT**: NVIDIA's specialized lowering passes

### Graph Compilation

This takes the intermediate "lowered" execution graph representation and generates the actual "low-level" code. This typically refers to the Triton kernels and other C++ implementations of operations.

Graph compilation is powerful and can do some impressive things:
- Based on the target hardware, pick the most optimized kernel available
- Experiment with different backends for operators: Aten, CUTLASS, Triton, Cpp, CuDNN, etc.
- Identify the best tiling strategy for operators like `matmul`
- Perform kernel fusion to reduce kernel launches and improve performance

During compilation, the optimization framework runs with different configurations and times the outputs. Finally it picks the fastest combinations and compiles that into machine-executable code (or a shared library if AOTI).

Graph compilation is done via tools like TorchInductor, TensorRT, or ONNX Runtime.

Inference optimization is highly hardware-dependent. When building an inference engine, ensure your runtime environment matches the exact specifications used during optimization, including PyTorch/TensorRT version, GPU model (e.g., A100), CUDA version, and other hardware-specific configurations.[^1]

[^1]: A mismatch between compilation and runtime environments is one of the most common sources of inference issues.

## Compilation Strategies

Deep learning inference optimization employs two primary compilation approaches:

**1. Static Compilation (Ahead-of-Time)**
- Compilation occurs before deployment
- Optimized for specific input shapes and configurations
- Generates a fixed, optimized runtime
- Well-suited for production environments with predictable workloads

**2. Dynamic Compilation (Just-in-Time)**
- Compilation happens during runtime
- Adapts to varying input shapes and patterns
- Includes on-the-fly autotuning
- Ideal for development and experimentation

### Comparison of Compilation Approaches

| Aspect | Static Compilation | Dynamic Compilation |
|--------|-------------------|-------------------|
| Compilation Time | Before deployment | During runtime |
| Performance | Consistent, predictable | Variable |
| Flexibility | Limited to predefined shapes | Adapts to varying inputs |
| Use Case | Production deployment | Development & experimentation |
| Initial Overhead | High (one-time) | Low (distributed) |
| Runtime Overhead | Minimal | Moderate |
| Hardware Optimization | Deep, hardware-specific | General, adaptable |
| Best For | Stable, production workloads | Dynamic, evolving models |
| Example | AOTInductor, TensorRT | `torch.compile` |

The choice between static and dynamic compilation depends on your specific requirements. Static compilation excels in production environments with predictable workloads, while dynamic compilation offers greater flexibility during development and for models with varying input patterns.

## Static Compilation

### AOTInductor (AOTI)

AOTI is a model optimization and compiler framework for deployment by PyTorch. The "AOT" in AOTInductor refers to "ahead-of-time". This is a static compilation technique, where we optimize our inference runtime before deployment and use the optimized version for productionizing. Though AOTI is very much in the early phase, it's a really good sign since it's torch native and opensourced!

```python
import os
import torch

class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(10, 16)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(16, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.sigmoid(x)

# Compile model for inference
with torch.no_grad():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SimpleModel().to(device)

    # Define example input with dynamic batch dimension
    example_inputs = (torch.randn(8, 10, device=device),)
    batch_dim = torch.export.Dim("batch", min=1, max=1024)

    # Compile model to shared library
    so_path = torch._export.aot_compile(
        model,
        example_inputs,
        dynamic_shapes={"x": {0: batch_dim}},
        options={"aot_inductor.output_path": os.path.join(os.getcwd(), "model.so")}
    )
```

### TensorRT Framework

TensorRT is an SDK by NVIDIA for high-performance deep learning inference, optimized specifically for NVIDIA GPUs. It's widely used in industry for production-grade deployments because of its speed and hardware-specific optimizations.

The core principle is the same as in any inference optimization framework just the toolings are a bit different:
- **Graph tracing & IR**: Export your model using ONNX
- **Graph compilation**: `trtexec` from TensorRT

TensorRT uses ONNX which is great since you can export your models from PyTorch or TensorFlow to ONNX which is a standardized representation.

#### Export to ONNX Format

```python
with torch.no_grad():
    torch.onnx.export(
        model.to(dtype).to(device),
        torch.randn((1, 15256, 4096), dtype=dtype, device=device),
        os.path.join(onnx_dir, "model.onnx"),
        input_names=["x"],
        output_names=["latents"],
        opset_version=19,
        dynamic_axes={
            "x": {1: "seq_len"},
            "latents": {1: "seq_len"}
        }
    )
```

The export process runs a forward pass with the provided input, capturing all operations in the execution path. You can visualize the resulting ONNX model using [Netron](https://netron.app/).

#### Generate TensorRT Engine

Once you have the ONNX model, use `trtexec` to create an optimized TensorRT engine:

```bash
/usr/src/tensorrt/bin/trtexec \
    --onnx="$export_dir/onnx/model.onnx" \
    --stronglyTyped \
    --timingCacheFile="$export_dir/timing.cache" \
    --optShapes=x:1x15256x4096 \
    --minShapes=x:1x15000x4096 \
    --maxShapes=x:1x16000x4096 \
    --saveEngine="$export_dir/model.engine" \
    --profilingVerbosity=detailed
```

**Key Parameters:**
- `optShapes`: Optimal input shapes for performance
- `minShapes`/`maxShapes`: Supported input dimension range
- `stronglyTyped`: Uses exact data types from ONNX file
- `timingCacheFile`: Caches optimization results for faster subsequent runs

## Dynamic Compilation with torch.compile

Unlike AOTI, `torch.compile` was designed for making your PyTorch code faster with minimal changes. It's a form of dynamic runtime optimization that sacrifices some control for ease of use.

### Step-by-Step Process

1. **TorchDynamo Tracing** — Intercepts Python code by tracing through the Python bytecode, identifies the computational graph and removes non-essential Python constructs

2. **AOTAutograd** — Splits the graph into forward and backward components, captures the autograd logic separately

3. **Backend Compilation** — The graph is passed to a backend (usually TorchInductor by default), which lowers the graph to lower-level IR and generates optimized code (Triton for GPU, C++ for CPU)

4. **Execution** — The compiled function is cached and reused for future calls

### Sample Code

```python
model = torch.compile(model, mode="reduce-overhead")
```

That's about it! It's super torch native and plays well with `torchao` and distributed computing.

### Compile Modes

| Mode | Autotune | CUDA Graphs | Use Case |
|------|----------|-------------|----------|
| `default` | Medium | Maybe | Balanced performance |
| `reduce-overhead` | Low | Yes | Fast for small batches |
| `max-autotune` | High | Yes | Max perf (can be unstable) |
| `max-autotune-no-cudagraphs` | High | No | Max tuning, safer for dynamic models |

## Further Reading

- [PyTorch AOTInductor Documentation](https://pytorch.org/docs/stable/torch.compiler_aot_inductor.html)
- [NVIDIA TensorRT](https://github.com/NVIDIA/TensorRT)
- [llm.c](https://github.com/karpathy/llm.c)

---
title: Understanding Inference Optimization Frameworks
date: 2025-04-04 08:57:00 +0530
categories: [Optimization]
tags: [ML, GPU]
math: false
pin: false
image:
  path: https://miro.medium.com/v2/resize:fit:1400/1*PiZ_bHavO0aWI-mPMyfASQ.png
  alt: Understanding Inference Optimization Frameworks
---


## Overview

Modern deep learning models require efficient execution to meet production demands. While the core logic of our code defines what we want to achieve, its execution depends on numerous lower-level components that translate our intentions into machine instructions. This translation layer, known as the runtime, is crucial for model inference performance.

Runtime optimization in deep learning focuses on improving the efficiency of model inference execution. As models grow larger and GPU resources remain expensive, optimizing runtime performance becomes essential for both cost savings and better user experience.

The default PyTorch runtime, while suitable for development and small-scale inference, includes significant overhead (as demonstrated by projects like [llm.c](https://github.com/karpathy/llm.c)). This overhead is inherent in its general-purpose design. For production-scale deployment, specialized optimization frameworks become necessary.

Modern inference optimization frameworks consist of two key components:
- Compiler: Transforms high-level model code into optimized low-level representations
- Runtime: Manages the execution of compiled code with minimal overhead

### Compiler Architecture

The compiler transforms high-level PyTorch model code into optimized low-level representations through a multi-stage process:

#### Graph Acquisition

Inference can be viewed as executing a sequence of operations to transform input data into predictions. While inputs may vary, the operation sequence remains constant (and often, so do the input shapes).

Graph acquisition converts this operation sequence into a computational graph where:
- Nodes represent individual operations
- Edges represent data flow and execution dependencies

Popular tools for graph acquisition include:
- Torch FX: PyTorch's native graph capture tool
- Torch Dynamo: Dynamic graph tracing for PyTorch
- ONNX: Cross-platform graph representation format

#### Graph Lowering

Graph lowering transforms the high-level computational graph into a lower-level intermediate representation (IR) that better matches the target hardware's execution model. This transformation enables:

- Direct mapping of PyTorch operations to hardware-specific kernels
- Removal of inference-irrelevant components (e.g., dropout, gradient computation)
- Static optimization of deterministic control flow
- Better optimization opportunities through hardware-aware transformations

Common lowering implementations include:
- AOTAutograd: Ahead-of-time autograd transformation
- FX passes: PyTorch's graph transformation passes
- ONNX Runtime: Cross-platform optimization passes
- TensorRT: NVIDIA's specialized lowering passes

#### Graph Compilation

This takes the intermediate "lowered" execution graph representation and generates the actual "low-level" code. This typically refers to the Triton kernels and other C++ impl of operations.

Graph compilation is really powerful and can do some very cool stuff like:
- Based on the target hardware, pick the most optimized kernel available
- Experiment with different backends for operators: Aten, CUTLASS, Triton, Cpp, CuDNN* etc
- Identify the best tiling strategy for operators like `matmul`
- Perform kernel fusion to reduce kernel launches and improve performance
- and many more ..

During compilation, the optimization framework runs with different configurations and times the outputs. Finally it picks the fastest combinations and compiles that into machine-executable code (or a shared library if AOTI).

Graph compilation is done via tools like TorchInductor or TensorRT or ONNX Runtime.

> Inference optimization is highly hardware-dependent. When building an inference engine, ensure your runtime environment matches the exact specifications used during optimization, including - PyTorch/TensorRT version, GPU model (e.g., A100), CUDA version, other hardware-specific configurations
{: .prompt-warning}

## Compilation Strategies

Deep learning inference optimization employs two primary compilation approaches:

1. **Static Compilation (Ahead-of-Time)**
   - Compilation occurs before deployment
   - Optimized for specific input shapes and configurations
   - Generates a fixed, optimized runtime
   - Well-suited for production environments with predictable workloads

2. **Dynamic Compilation (Just-in-Time)**
   - Compilation happens during runtime
   - Adapts to varying input shapes and patterns
   - Includes on-the-fly autotuning
   - Ideal for development and experimentation

### Comparison of Compilation Approaches

| Aspect | Static Compilation | Dynamic Compilation |
|--------|-------------------|-------------------|
| **Compilation Time** | Before deployment | During runtime |
| **Performance** | Consistent, predictable so optimization are aggressive | Variable so optimizations are conservative |
| **Flexibility** | Limited to predefined shapes | Adapts to varying inputs |
| **Use Case** | Production deployment | Development & experimentation |
| **Initial Overhead** | High (one-time) | Low (distributed) |
| **Runtime Overhead** | Minimal | Moderate |
| **Hardware Optimization** | Deep, hardware-specific | General, adaptable |
| **Best For** | Stable, production workloads | Dynamic, evolving models |
| **Example** | AOTInductor, TensorRT | `torch.compile` |

> **Note**: The choice between static and dynamic compilation depends on your specific requirements. Static compilation excels in production environments with predictable workloads, while dynamic compilation offers greater flexibility during development and for models with varying input patterns.
{: .prompt-tip}

## Static Compilation

### AOTInductor (AOTI)

AOTI is a model optimization and compiler framework for deployment by PyTorch. The "AOT" in AOTInductor refers to "ahead-of-time". This is a static compilation technique, where we optimize our inference runtime before deployment and use the optimized version for productionizing. Though AOTI is very much in the early phase, it's a really good sign since it's torch native and opensourced!

####  Sample Code

> Quick disclaimer. This is definitely an over-simplification and in a more practical scenario, the ride's going to be super bumpy since it's still reaching maturity. But I feel this captures the overall stucture and flow.
{: .prompt-info}

> **Note**: This example demonstrates the basic structure using PyTorch 2.4.0/2.5.1. The API is subject to change as the framework matures.
{: .prompt-warning}

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

The core principle is the same as in any inference optimization framework just the toolings are a bit different.
- Graph tracing & IR - Export your model using onnx
- Graph compilation - `trtexec` from TensorRT

TensorRT uses ONNX which is great since you can export your models from PyTorch or TensorFlow to ONNX which is a standardized representation. 

#### Export to ONNX Format

TensorRT requires models to be exported in ONNX format first. Here's how to export a PyTorch model:

```python
with torch.no_grad():
    torch.onnx.export(
        model.to(dtype).to(device),  # Your PyTorch model
        torch.randn((1, 15256, 4096), dtype=dtype, device=device),  # Example input
        os.path.join(onnx_dir, "model.onnx"),  # Output path
        input_names=["x"],  # Input tensor names
        output_names=["latents"],  # Output tensor names
        opset_version=19,  # ONNX operator set version
        dynamic_axes={
            "x": {1: "seq_len"},  # Dynamic sequence length
            "latents": {1: "seq_len"}
        }
    )
```

> **Note**: The export process runs a forward pass with the provided input, capturing all operations in the execution path. You can visualize the resulting ONNX model using [Netron](https://netron.app/).
{: .prompt-info}

> **Tip**: For large models, export without weights to create a smaller, more manageable visualization file (typically in KBs).
{: .prompt-tip}

#### Generate TensorRT Engine

Once you have the ONNX model, use `trtexec` to create an optimized TensorRT engine:

```bash
/usr/src/tensorrt/bin/trtexec \
    --onnx="$export_dir/onnx/model.onnx" \
    --stronglyTyped \
    --timingCacheFile="$export_dir/v104/timing_model.cache" \
    --optShapes=x:1x15256x4096 \
    --minShapes=x:1x15000x4096 \
    --maxShapes=x:1x16000x4096 \
    --separateProfileRun \
    --saveEngine="$export_dir/v104/model.engine" \
    --exportTimes="$export_dir/v104/times_model.json" \
    --exportProfile="$export_dir/v104/profile_model.json" \
    --exportLayerInfo="$export_dir/v104/layerinfo_model.json" \
    --profilingVerbosity=detailed | tee "$export_dir/v104/model.log"
```

#### Key Parameters Explained

1. **Shape Parameters**
   - `optShapes`: Optimal input shapes for performance
   - `minShapes`: Minimum supported input dimensions
   - `maxShapes`: Maximum supported input dimensions
   - Supports multiple inputs and optimization profiles

2. **Type Handling**
   - `stronglyTyped`: Uses exact data types from ONNX file
   - Can be overridden for specific precision (e.g., `float16`, `bfloat16`)

3. **Profiling and Debugging**
   - `timingCacheFile`: Caches optimization results for faster subsequent runs
   - `profilingVerbosity`: Controls detail level of performance profiling
   - Various export options for analyzing engine behavior

## Dynamic Compilation with `torch.compile`

Unlike AOTI, `torch.compile` was designed for making your pytorch code faster with minimal changes. Also `torch.compile` is a form of dynamic runtime optimization. And as expected, this comes with a few tradeoffs. We sacrifice control and fine-grained control for the ease of use. `torch.compile` allows us to experiment & be more flexible and while AOTI makes sense for a more production ready usecase.

Few things to note though:

- `torch.compile` actually **uses AOTAutograd** internally **as part of its stack**. So you can think of `torch._export.aot_compile` as being a **lower-layer primitive** that `torch.compile` builds upon.

- If you're trying to plug into the compilation stack with **custom compilers or transformations**, you'll want to use `aot_compile`.

### Step-by-Step Process

- **TorchDynamo Tracing**
    - Intercepts Python code by tracing through the Python bytecode.
    - Identifies the computational graph and removes non-essential Python constructs.
        
- **AOTAutograd (Ahead-of-Time Autograd)**
    - Splits the graph into forward and backward components.
    - Captures the autograd logic separately, which helps in further optimization.
        
- **Backend Compilation**
    - The graph is then passed to a backend (usually TorchInductor by default).
    - TorchInductor lowers the graph to lower-level IR (intermediate representation) and generates optimized code:
        - Triton for GPU
        - C++ for CPU
            
- **Execution**
    - The compiled function is cached and reused for future calls.
    - Supports dynamic shapes and symbolic dimensions if specified.

So under the hood, `torch.compile` actually builds this little pipeline each time it hits a new graph. That's where some overhead comes from initially, but later invocations are super fast.

### Sample Code

```python
model = torch.compile(model, mode="reduce-overhead")
```

Yup that's about it!! It's super torch native and plays well with `torchao` and distributed computing (unlike AOTI which is more suited for single GPU tasks).

## Compile "modes"

Can be either "default", "reduce-overhead", "max-autotune" or "max-autotune-no-cudagraphs"

- "`default`" is the default mode, which is a good balance between performance and overhead
    
- "`reduce-overhead`" is a mode that reduces the overhead of python with CUDA graphs, useful for small batches. Reduction of overhead can come at the cost of more memory usage, as we will cache the workspace memory required for the invocation so that we do not have to reallocate it on subsequent runs. Reduction of overhead is not guaranteed to work; today, we only reduce overhead for CUDA only graphs which do not mutate inputs. 
    
- "`max-autotune`" is a mode that leverages Triton or template based matrix multiplications on supported devices and Triton based convolutions on GPU. It enables CUDA graphs by default on GPU.
    
- "`max-autotune-no-cudagraphs`" is a mode similar to "max-autotune" but without CUDA graphs

| Mode                       | Autotune | CUDA Graphs | Use Case                                     |
|----------------------------|----------|-------------|----------------------------------------------|
| default                    | Medium   | Maybe       | Balanced performance                         |
| reduce-overhead            | Low      | Yes         | Fast for small batches                       |
| max-autotune               | High     | Yes         | Max perf (can be unstable)                   |
| max-autotune-no-cudagraphs | High     | ❌ No        | Max tuning, safer for dynamic/complex models |


## Further Readings

- https://pytorch.org/docs/stable/torch.compiler_aot_inductor.html
- https://pytorch.org/tutorials/recipes/torch_export_aoti_python.html
- https://github.com/NVIDIA/TensorRT
- https://github.com/karpathy/llm.c
- https://docs.google.com/document/d/1y5CRfMLdwEoF1nTk9q8qEu1mgMUuUtvhklPKJ2emLU8/edit?tab=t.0#heading=h.ivdr7fmrbeab

---
layout: post
title: "Building Custom TensorRT Plugins"
date: 2024-12-13
author: "Allen Philip J"
description: "A practical guide to extending TensorRT with custom plugins, using FlashAttention 3 as an implementation example."
tags: [TensorRT, GPU, Optimization]
katex: true
---

TensorRT's standard operations cover many common use cases, but there are scenarios where custom solutions become necessary:

1. **Third-party Integration** — When working with specialized libraries that lack direct PyTorch equivalents, or cases where tracing through external dependencies isn't possible.

2. **Complex Control Flow** — Models with intricate conditional logic that can't be simplified, or dynamic execution paths that don't map well to static compilation.

3. **Python Integration** — Situations requiring Python execution within the TensorRT engine, or custom operations that benefit from Python's flexibility.

In this guide, we'll explore how to implement a custom attention mechanism using FlashAttention 3 as a practical example. This will demonstrate the process of extending TensorRT's capabilities while maintaining performance and compatibility.

## Supporting Custom TRT Plugins

Sometimes standard operations aren't sufficient. This could be when you're using certain third party libraries that don't have a torch counterpart for us to allow for tracing. Or say you have a block of code with a lot of conditional flow and can't replace it with a hardcoded version. Basically any scenario when you'd like to fall back to a python execution from within a TensorRT engine.

Let's have a look at how to do this for building a TRT engine which supports a custom attention kernel like FlashAttention 3.

## Step 1: Define Custom ONNX Operator

First, we need to define a custom operator that PyTorch can trace and export to ONNX:

```python
from typing import Sequence

import torch
from torch._custom_op import impl as custom_op
from torch.onnx import symbolic_helper
from torch.onnx._internal import jit_utils

from .fa3 import FA3

@custom_op.custom_op("attn::custom_attn")
def custom_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mode: Sequence[int],
) -> torch.Tensor: ...


@custom_attn.impl("cpu")
def custom_attn_cpu(q, k, v, mode) -> torch.Tensor:
    return _attn(q, k, v, mode[0])


@custom_attn.impl("cuda")
def custom_attn_cuda(q, k, v, mode) -> torch.Tensor:
    return _attn(q, k, v, mode[0])


@symbolic_helper.parse_args("v", "v", "v", "is")
def symbolic_custom_attn(g: jit_utils.GraphContext, q, k, v, mode) -> torch.Value:
    return g.op(
        "attn::CustomAttnPlugin",
        q,
        k,
        v,
        mode_i=mode,
        outputs=1,  # returns tuple if != 1
    ).setTypeAs(q)


def _attn(q, k, v, attn_mode=0):
    # FA3 Attn
    dtype_in = q.dtype
    if attn_mode == 1:
        dtype = torch.float8_e4m3fn
        q = q.contiguous().to(dtype)
        k = k.contiguous().to(dtype)
        v = v.contiguous().to(dtype)
    return FA3.apply(q, k, v).to(dtype_in)
```

## Step 2: Register TensorRT Plugin

Now that TRT can identify this is a new operator it hasn't seen, we need to map it to a PyTorch/C++ operation. This is done by writing the `CustomTRTPlugin` class.[^1]

[^1]: Reference: [Official NVIDIA TensorRT Python Plugin Sample](https://github.com/NVIDIA/TensorRT/blob/c8a50438f6929470800c22088480784b254a7ac0/samples/python/python_plugin/circ_pad_plugin_torch.py)

**Keep in mind:**

- Use consistent TRT names and arguments across ONNX and TRT
- Be careful when adding more args and handle the dtypes carefully
- The method of interest is `enqueue()` which actually calls the Python code

```python
from typing import Any
import cupy as cp
import numpy as np
import tensorrt as trt
import torch
from cuda.bindings.driver import cuMemcpyDtoDAsync
from polygraphy.json import from_json, to_json

from .onnx_op import _attn


def volume(d: trt.Dims) -> np.ndarray:
    return np.prod(d)


class CustomAttnPlugin(trt.IPluginV2DynamicExt):
    def __init__(self, fc=None) -> None:
        trt.IPluginV2DynamicExt.__init__(self)
        self.num_outputs = 1
        self.plugin_namespace = ""
        self.plugin_type = "CustomAttnPlugin"
        self.plugin_version = "1"

        if fc is not None:
            assert fc[0].name == "mode"
            self.mode = int(fc[0].data[0])

    def get_output_datatype(self, index, input_types) -> Any:
        return input_types[0]

    def get_output_dimensions(self, output_index, inputs, exprBuilder) -> Any:
        output_dims = trt.DimsExprs(inputs[0])
        return output_dims

    def serialize(self) -> Any:
        return to_json({"mode": self.mode})

    def configure_plugin(self, inp, out) -> None:
        Q_dims = inp[0].desc.dims
        self.Q_shape = np.zeros((len(Q_dims),))
        for i in range(len(Q_dims)):
            self.Q_shape[i] = Q_dims[i]

        K_dims = inp[1].desc.dims
        self.K_shape = np.zeros((len(K_dims),))
        for i in range(len(K_dims)):
            self.K_shape[i] = K_dims[i]

        V_dims = inp[2].desc.dims
        self.V_shape = np.zeros((len(Q_dims),))
        for i in range(len(V_dims)):
            self.V_shape[i] = Q_dims[i]

    def supports_format_combination(self, pos, in_out, num_inputs) -> bool:
        assert num_inputs == 3
        assert pos < len(in_out)

        desc = in_out[pos]
        if desc.format != trt.TensorFormat.LINEAR:
            return False

        # first input should be float16 or float32
        if pos == 0:
            return bool(
                desc.type == trt.DataType.FLOAT
                or desc.type == trt.DataType.HALF
                or desc.type == trt.DataType.BF16
            )
        if pos == 1:
            return bool(
                desc.type == trt.DataType.FLOAT
                or desc.type == trt.DataType.HALF
                or desc.type == trt.DataType.BF16
            )
        if pos == 2:
            return bool(
                desc.type == trt.DataType.FLOAT
                or desc.type == trt.DataType.HALF
                or desc.type == trt.DataType.BF16
            )

        # output should have the same type as the input
        if pos == 3:
            return bool(in_out[0].type == desc.type)

        return False

    def enqueue(self, input_desc, output_desc, inputs, outputs, workspace, stream) -> int:
        # Do manual copy for BF16 as numpy doesn't support it
        if input_desc[0].type == trt.DataType.BF16:
            inp_dtype = np.uint16
            q_mem = cp.cuda.UnownedMemory(
                inputs[0], volume(input_desc[0].dims) * cp.dtype(inp_dtype).itemsize, self
            )
            k_mem = cp.cuda.UnownedMemory(
                inputs[1], volume(input_desc[1].dims) * cp.dtype(inp_dtype).itemsize, self
            )
            v_mem = cp.cuda.UnownedMemory(
                inputs[2], volume(input_desc[2].dims) * cp.dtype(inp_dtype).itemsize, self
            )
            c_mem = cp.cuda.UnownedMemory(
                outputs[0],
                volume(output_desc[0].dims) * cp.dtype(inp_dtype).itemsize,
                self,
            )

            q_ptr = cp.cuda.MemoryPointer(q_mem, 0)
            k_ptr = cp.cuda.MemoryPointer(k_mem, 0)
            v_ptr = cp.cuda.MemoryPointer(v_mem, 0)
            c_ptr = cp.cuda.MemoryPointer(c_mem, 0)

            c_d = cp.ndarray((volume(output_desc[0].dims)), dtype=inp_dtype, memptr=c_ptr)

            q_t = torch.empty(tuple(input_desc[0].dims), device="cuda", dtype=torch.bfloat16)
            k_t = torch.empty(tuple(input_desc[1].dims), device="cuda", dtype=torch.bfloat16)
            v_t = torch.empty(tuple(input_desc[2].dims), device="cuda", dtype=torch.bfloat16)

            cuMemcpyDtoDAsync(q_t.data_ptr(), q_ptr, q_t.nbytes, stream)
            cuMemcpyDtoDAsync(k_t.data_ptr(), k_ptr, k_t.nbytes, stream)
            cuMemcpyDtoDAsync(v_t.data_ptr(), v_ptr, v_t.nbytes, stream)

            out = torch.reshape(_attn(q_t, k_t, v_t), (-1,))

            cuMemcpyDtoDAsync(c_d.data.ptr, out.data_ptr(), out.nbytes, stream)
        else:
            inp_dtype = trt.nptype(input_desc[0].type)

            q_mem = cp.cuda.UnownedMemory(
                inputs[0], volume(input_desc[0].dims) * cp.dtype(inp_dtype).itemsize, self
            )
            k_mem = cp.cuda.UnownedMemory(
                inputs[1], volume(input_desc[1].dims) * cp.dtype(inp_dtype).itemsize, self
            )
            v_mem = cp.cuda.UnownedMemory(
                inputs[2], volume(input_desc[2].dims) * cp.dtype(inp_dtype).itemsize, self
            )
            c_mem = cp.cuda.UnownedMemory(
                outputs[0],
                volume(output_desc[0].dims) * cp.dtype(inp_dtype).itemsize,
                self,
            )

            q_ptr = cp.cuda.MemoryPointer(q_mem, 0)
            k_ptr = cp.cuda.MemoryPointer(k_mem, 0)
            v_ptr = cp.cuda.MemoryPointer(v_mem, 0)
            c_ptr = cp.cuda.MemoryPointer(c_mem, 0)

            q_d = cp.ndarray(tuple(input_desc[0].dims), dtype=inp_dtype, memptr=q_ptr)
            k_d = cp.ndarray(tuple(input_desc[1].dims), dtype=inp_dtype, memptr=k_ptr)
            v_d = cp.ndarray(tuple(input_desc[2].dims), dtype=inp_dtype, memptr=v_ptr)
            c_d = cp.ndarray((volume(output_desc[0].dims)), dtype=inp_dtype, memptr=c_ptr)

            q_t = torch.as_tensor(q_d, device="cuda")
            k_t = torch.as_tensor(k_d, device="cuda")
            v_t = torch.as_tensor(v_d, device="cuda")

            out = _attn(q_t, k_t, v_t, self.mode)

            cp.copyto(c_d, cp.reshape(cp.asarray(out), (-1,)))
        return 0

    def clone(self) -> trt.IPluginV2DynamicExt:
        cloned_plugin = CustomAttnPlugin()
        cloned_plugin.__dict__.update(self.__dict__)
        return cloned_plugin


class CustomAttnPluginCreator(trt.IPluginCreator):
    def __init__(self) -> None:
        trt.IPluginCreator.__init__(self)
        self.name = "CustomAttnPlugin"
        self.plugin_namespace = ""
        self.plugin_version = "1"
        self.field_names = trt.PluginFieldCollection(
            [trt.PluginField("mode", np.array([]), trt.PluginFieldType.INT32)]
        )

    def create_plugin(self, name, fc) -> trt.IPluginV2DynamicExt:
        return CustomAttnPlugin(fc)

    def deserialize_plugin(self, name, data) -> trt.IPluginCreator:
        j = dict(from_json(data.decode("utf-8")))
        deserialized: trt.IPluginV2DynamicExt = CustomAttnPlugin()
        deserialized.__dict__.update(j)
        return deserialized
```

## Step 3: Building the TensorRT Engine

Building a TensorRT engine with custom plugins requires special consideration:

1. **Plugin Registration** — Direct use of `trtexec` is not possible due to plugin registration requirements. The engine must be built programmatically through PyTorch to ensure proper plugin initialization before engine creation.

2. **ONNX Operator Naming** — ONNX export may modify operator names for repeated operations. The `rename_custom_op()` function ensures consistent naming to maintain compatibility between ONNX and TensorRT representations.[^2]

[^2]: The custom plugin must be registered before the TensorRT engine is built to ensure proper functionality.

```python
import tensorrt as trt
from polygraphy.backend.trt import (
    CreateConfig,
    EngineFromNetwork,
    NetworkFromOnnxPath,
    Profile,
    save_engine,
    TrtRunner,
    EngineFromBytes,
)

def rename_custom_op(onnx_path):
    """
    Rename the custom op for TRT plugin compatibility
    """
    print(f"Loading ONNX model from {onnx_path}...")
    model_onnx = onnx.load(onnx_path)
    graph = gs.import_onnx(model_onnx)

    for node in graph.nodes:
        if node.op == "CustomAttnPlugin":
            print("Found CustomAttnPlugin node...")
            print(node)
            node.name = "CustomAttnPlugin"
            node.op = "CustomAttnPlugin"

    print("Exporting the graph...")
    graph.toposort()
    graph.fold_constants()
    graph.cleanup()
    model_onnx = gs.export_onnx(graph)
    onnx_path = onnx_path.split("/model.onnx")[0] + "/model_mod.onnx"
    onnx.save(
        model_onnx,
        onnx_path,
        save_as_external_data=True,
        location=os.path.basename(onnx_path) + "_data",
    )
    print(f"ONNX model '{onnx_path}' saved successfully.")
    return onnx_path


# Register plugin creator
print("Registering TRT plugin...")
plg_registry = trt.get_plugin_registry()
my_plugin_creator = CustomAttnPluginCreator()
plg_registry.register_creator(my_plugin_creator, "")

# Register custom op to ONNX
torch.onnx.register_custom_op_symbolic("attn::custom_attn", symbolic_custom_attn, 1)

# Rename custom op correctly
rename_custom_op(f"{onnx_dir}/model.onnx")

# Build the engine
print("Building TRT engine...")
profiles = [
    Profile().add(
        "x",
        min=(1, 15000, 4096),
        opt=(1, 15708, 4096),
        max=(1, 16500, 4096),
    ),
]
build_engine = EngineFromNetwork(
    NetworkFromOnnxPath(str(f"{onnx_dir}/model_mod.onnx"), strongly_typed=True),
    CreateConfig(profiles=profiles),
)

# Save the engine
print("Saving TRT engine...")
with build_engine() as engine:
    save_engine(engine, path=str(trt_path))
print(f"Engine saved to {trt_path}")
```

## Step 4: Using the TRT Engine for Inference

We can use the `tensorrt` Python package for running inference on the generated TRT engine. But keep in mind this is just a Python wrapper on a C/C++ runtime. Hence a lot of overhead like managing memory falls on us to take care of.

Here's a wrapper class for inference:

```python
from pathlib import Path
import tensorrt as trt
import torch
from cuda import cudart

from .plugins.fa3_plugin import CustomAttnPluginCreator


TRT_DTYPE_TO_TORCH = {
    trt.float32: torch.float32,
    trt.float16: torch.float16,
    trt.int32: torch.int32,
    trt.int64: torch.int64,
    trt.int8: torch.int8,
    trt.bool: torch.bool,
    trt.bfloat16: torch.bfloat16,
}


class TRTEngine:
    def __init__(
        self,
        engine: trt.ICudaEngine,
        device: torch.device,
        profile_idx: int | None = None
    ) -> None:
        self.inputs: dict[str, torch.Tensor] = {}
        self.inputs_bind: dict[str, int] = {}
        self.outputs: dict[str, torch.Tensor] = {}
        self.bindings: list[int] = []

        self.stream = torch.cuda.current_stream(device=device).cuda_stream
        self.engine = engine
        self.context = engine.create_execution_context()
        self.profile_idx = profile_idx
        self.device = device

        self._allocate_buffers()

    @classmethod
    def from_trt(
        cls,
        trt_file_path: Path,
        device: torch.device,
        profile_idx: int | None = None
    ) -> "TRTEngine":
        runtime = trt.Runtime(trt.Logger())
        cudart.cudaSetDevice(device.index)

        # Register the custom plugin
        plg_registry = trt.get_plugin_registry()
        my_plugin_creator = CustomAttnPluginCreator()
        plg_registry.register_creator(my_plugin_creator, "")

        print(f"Reading TRT engine from {trt_file_path} on device {device}")
        with open(trt_file_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
            if engine is None:
                raise RuntimeError(f"Failed to load TRT engine from {trt_file_path}")
        return cls(engine, device=device, profile_idx=profile_idx)

    def _allocate_buffers(self) -> None:
        profile_idx = self.profile_idx
        engine = self.engine
        input_shape: tuple[int, ...] | None = None

        with torch.cuda.device(device=self.device):
            for i in range(engine.num_io_tensors):
                tensor_name = engine.get_tensor_name(i)
                if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                    input_shape = tuple(
                        engine.get_tensor_shape(tensor_name)
                        if profile_idx is None
                        else engine.get_tensor_profile_shape(tensor_name, profile_idx)[-1]
                    )
                    dtype = engine.get_tensor_dtype(tensor_name)
                    tensor = torch.empty(
                        input_shape,
                        dtype=TRT_DTYPE_TO_TORCH[dtype],
                        device=self.device
                    )
                    self.inputs[tensor_name] = tensor
                    self.inputs_bind[tensor_name] = i
                else:
                    assert input_shape is not None
                    output_shape = tuple(engine.get_tensor_shape(tensor_name))
                    if -1 in output_shape:
                        index = output_shape.index(-1)
                        output_shape = list(output_shape)
                        output_shape[index] = input_shape[index]
                        output_shape = tuple(output_shape)
                    dtype = engine.get_tensor_dtype(tensor_name)
                    tensor = torch.empty(
                        output_shape,
                        dtype=TRT_DTYPE_TO_TORCH[dtype],
                        device=self.device
                    )
                    self.outputs[tensor_name] = tensor

                self.context.set_tensor_address(tensor_name, tensor.data_ptr())

    def execute(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        with torch.cuda.device(device=self.device):
            return self.execute_on_cuda(inputs)

    def execute_on_cuda(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        for key, value in inputs.items():
            self.inputs[key][:, :int(value.shape[1]), :].copy_(value)
            if key not in self.inputs_bind:
                raise KeyError(f"Invalid input name for TRT engine: {key}")

            self.context.set_input_shape(key, tuple(value.shape))
            self.context.set_tensor_address(key, value.data_ptr())

        self.context.execute_async_v3(stream_handle=self.stream)
        return self.outputs


def execute_trt(engine: TRTEngine, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    outputs = engine.execute_on_cuda(inputs)
    input_key = next(iter(inputs))
    input_seqlen = inputs[input_key].shape[1]

    for output_key in outputs:
        outputs[output_key] = outputs[output_key][:, :input_seqlen, :]

    res = list(outputs.values())
    return res if len(res) > 1 else res[0]


class TrtEngineCallable(torch.nn.Module):
    def __init__(self, trt_engine: TRTEngine):
        super().__init__()
        self.trt_engine = trt_engine

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return execute_trt(self.trt_engine, inputs={"x": x})


if __name__ == '__main__':
    trt_path = "/path/to/engine.trt"
    trt_engine = TRTEngine.from_trt(trt_path, torch.device("cuda:0"), 0)

    trt_fn = TrtEngineCallable(trt_engine)

    # Sample inference
    output = trt_fn(torch.randn(1, 15255, 4096, dtype=torch.bfloat16, device="cuda:0"))
```

Don't forget to register the custom plugin before loading the engine:[^3]

[^3]: This registration step is required every time you load a serialized engine that uses custom plugins.

```python
plg_registry = trt.get_plugin_registry()
my_plugin_creator = CustomAttnPluginCreator()
plg_registry.register_creator(my_plugin_creator, "")
```

---
title: Building Custom TensorRT Plugins
date: 2025-04-13 12:45:00 +0530
categories: [Optimization]
tags: [ML, GPU]
math: false
pin: false
image:
  path: https://www.hostinger.com/tutorials/wp-content/uploads/sites/2/2019/01/what-is-wordpress-plugin-1.webp
  alt: Building Custom TensorRT Plugins
---

## Extending TensorRT with Custom Plugins

TensorRT's standard operations cover many common use cases, but there are scenarios where custom solutions become necessary:

1. **Third-party Integration**
   - When working with specialized libraries that lack direct PyTorch equivalents
   - Cases where tracing through external dependencies isn't possible

2. **Complex Control Flow**
   - Models with intricate conditional logic that can't be simplified
   - Dynamic execution paths that don't map well to static compilation

3. **Python Integration**
   - Situations requiring Python execution within the TensorRT engine
   - Custom operations that benefit from Python's flexibility

In this guide, we'll explore how to implement a custom attention mechanism using FlashAttention 3 as a practical example. This will demonstrate the process of extending TensorRT's capabilities while maintaining performance and compatibility.

## Supporting Custom TRT Plugins

Sometimes standard operations aren't sufficient. This could be when you're using certain thrid party libraries that don't a torch counterpart for us to allow for tracing. Or say you have block of code with a lot of conditional flow and can't replace it with a hardcoded version. Basically any scenario when you'd like to fall back to a python execution from within a TensorRT engine.

Let's have a look how to do this for building a TRT engine which supports a custom attention kernel like say FlashAttention 3.

### Define Custom ONNX Operator
   
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
    # attn_scale: Sequence[float],
    # window_size: Sequence[int],
) -> torch.Tensor: ...


@custom_attn.impl("cpu")
def custom_attn_cpu(q, k, v, mode) -> torch.Tensor:
    return _attn(q, k, v, mode[0])


@custom_attn.impl("cuda")
def custom_attn_cuda(q, k, v, mode) -> torch.Tensor:
    return _attn(q, k, v, mode[0])


@symbolic_helper.parse_args("v", "v", "v", "is")
def symbolic_custom_attn(g: jit_utils.GraphContext, q, k, v, mode,) -> torch.Value:
    return g.op(
        "attn::CustomAttnPlugin",
        q,
        k,
        v,
        mode_i=mode,
        outputs=1,  # returns tuple if != 1
    ).setTypeAs(q)

def _attn(
    q,
    k,
    v,
    attn_mode=0, # default to bf16/fp16 attn
):
    # FA3 Attn
    dtype_in = q.dtype
    if attn_mode == 1:
        dtype = torch.float8_e4m3fn

        q = q.contiguous().to(dtype)
        k = k.contiguous().to(dtype)
        v = v.contiguous().to(dtype)
    return FA3.apply(q, k, v).to(dtype_in)
```

### Register TensorRT Plugin

Now that TRT can identify that this is a new operator it has not seen, we need to map it to a pytorch/C++ operation. This is done by writing the `CustomTRTPlugin` class as show below.

> Reference Official: https://github.com/NVIDIA/TensorRT/blob/c8a50438f6929470800c22088480784b254a7ac0/samples/python/python_plugin/circ_pad_plugin_torch.py
{: .prompt-info}

**Keep in mind to:**
- Use consistent TRT names and arguments all across onnx and trt
- Be careful on adding more args and handle the dtypes carefully
- Keep an eyes on the method of interest `enqueue(..)` which actually calls the Python code


```python
from typing import Any
import cupy as cp
import numpy as np
import tensorrt as trt
import torch
from cuda.bindings.driver import cuMemcpyDtoDAsync
from polygraphy.json import from_json, to_json

from .onnx_op import _attn


def volume(d: trt.Dims) -> np.ndarray:  # type: ignore[type-arg]
    return np.prod(d)  # type: ignore[no-any-return]


class CustomAttnPlugin(trt.IPluginV2DynamicExt):  # type: ignore[misc]
    def __init__(self, fc=None) -> None:  # type: ignore[no-untyped-def]
        trt.IPluginV2DynamicExt.__init__(self)
        self.num_outputs = 1
        self.plugin_namespace = ""
        self.plugin_type = "CustomAttnPlugin"
        self.plugin_version = "1"

        if fc is not None:
            assert fc[0].name == "mode"
            self.mode = int(fc[0].data[0])

    def get_output_datatype(self, index, input_types) -> Any:  # type: ignore[no-untyped-def]
        return input_types[0]

    def get_output_dimensions(self, output_index, inputs, exprBuilder) -> Any:  # type: ignore[no-untyped-def]
        output_dims = trt.DimsExprs(inputs[0])
        return output_dims

    def serialize(self) -> Any:
        return to_json({"mode": self.mode})

    def configure_plugin(self, inp, out) -> None:  # type: ignore[no-untyped-def]
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

    def supports_format_combination(self, pos, in_out, num_inputs) -> bool:  # type: ignore[no-untyped-def]
        assert num_inputs == 3
        assert pos < len(in_out)

        desc = in_out[pos]
        if desc.format != trt.TensorFormat.LINEAR:
            return False

        # first input should be float16 or float32
        if pos == 0:
            return bool(
                desc.type == trt.DataType.FLOAT or desc.type == trt.DataType.HALF or desc.type == trt.DataType.BF16
            )
        if pos == 1:
            return bool(
                desc.type == trt.DataType.FLOAT or desc.type == trt.DataType.HALF or desc.type == trt.DataType.BF16
            )
        if pos == 2:
            return bool(
                desc.type == trt.DataType.FLOAT or desc.type == trt.DataType.HALF or desc.type == trt.DataType.BF16
            )

        # output should have the same type as the input
        if pos == 3:
            return bool(in_out[0].type == desc.type)

        return False

    def enqueue(self, input_desc, output_desc, inputs, outputs, workspace, stream) -> int:  # type: ignore[no-untyped-def]
        # Do manual copy for BF16 as numpy doesn't support it
        if input_desc[0].type == trt.DataType.BF16:
            inp_dtype = np.uint16
            q_mem = cp.cuda.UnownedMemory(inputs[0], volume(input_desc[0].dims) * cp.dtype(inp_dtype).itemsize, self)
            k_mem = cp.cuda.UnownedMemory(inputs[1], volume(input_desc[1].dims) * cp.dtype(inp_dtype).itemsize, self)
            v_mem = cp.cuda.UnownedMemory(inputs[2], volume(input_desc[2].dims) * cp.dtype(inp_dtype).itemsize, self)
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

            q_mem = cp.cuda.UnownedMemory(inputs[0], volume(input_desc[0].dims) * cp.dtype(inp_dtype).itemsize, self)
            k_mem = cp.cuda.UnownedMemory(inputs[1], volume(input_desc[1].dims) * cp.dtype(inp_dtype).itemsize, self)
            v_mem = cp.cuda.UnownedMemory(inputs[2], volume(input_desc[2].dims) * cp.dtype(inp_dtype).itemsize, self)
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


class CustomAttnPluginCreator(trt.IPluginCreator):  # type: ignore[misc]
    def __init__(self) -> None:
        trt.IPluginCreator.__init__(self)
        self.name = "CustomAttnPlugin"
        self.plugin_namespace = ""
        self.plugin_version = "1"
        self.field_names = trt.PluginFieldCollection([trt.PluginField("mode", np.array([]), trt.PluginFieldType.INT32)])

    def create_plugin(self, name, fc) -> trt.IPluginV2DynamicExt:  # type: ignore[no-untyped-def]
        return CustomAttnPlugin(fc)

    def deserialize_plugin(self, name, data) -> trt.IPluginCreator:  # type: ignore[no-untyped-def]
        j = dict(from_json(data.decode("utf-8")))
        deserialized: trt.IPluginV2DynamicExt = CustomAttnPlugin()
        deserialized.__dict__.update(j)
        return deserialized
```

### Building and Loading the TensorRT Engine

Building a TensorRT engine with custom plugins requires special consideration:

1. **Plugin Registration**
   - Direct use of `trtexec` is not possible due to plugin registration requirements
   - The engine must be built programmatically through PyTorch
   - This ensures proper plugin initialization before engine creation

2. **ONNX Operator Naming**
   - ONNX export may modify operator names for repeated operations
   - The `rename_custom_op()` function ensures consistent naming
   - This maintains compatibility between ONNX and TensorRT representations

> **Note**: The custom plugin must be registered before the TensorRT engine is built to ensure proper functionality.
{: .prompt-info}


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
    print(f"!!! Loading ONNX model from {onnx_path} ... ")
    model_onnx = onnx.load(onnx_path)
    graph = gs.import_onnx(model_onnx)
    for node in graph.nodes:
        if node.op == "CustomAttnPlugin":
            print("!!! Found CustomAttnPlugin node ... ")
            print(node)
            node.name = "CustomAttnPlugin"
            node.op = "CustomAttnPlugin"

    print("!!! Exporting the graph ... ")
    graph.toposort()
    graph.fold_constants()
    graph.cleanup()
    model_onnx = gs.export_onnx(graph)
    onnx_path = onnx_path.split("/clio4.onnx")[0] + "/clio4_mod.onnx"
    onnx.save(
        model_onnx,
        onnx_path,
        save_as_external_data=True,
        location=os.path.basename(onnx_path) + "_data",
    )
    print(f"ONNX model '{onnx_path}' saved successfully.")
    return onnx_path

# Register plugin creator
print("!!! Registering TRT plugin ... ")
plg_registry = trt.get_plugin_registry()
my_plugin_creator = CustomAttnPluginCreator()
plg_registry.register_creator(my_plugin_creator, "")

# Register custom op to onnx
torch.onnx.register_custom_op_symbolic("attn::custom_attn", symbolic_custom_attn, 1)

# Rename custom op correctly
rename_custom_op(f"{onnx_in_dir}/clio4.onnx")

print("!!! Building TRT engine for block_in ... ")
profiles = [
    Profile().add(
        "x",
        min=(1, 15000, 4096),
        opt=(1, 15708, 4096),
        max=(1, 16500, 4096),
    ),
]
build_engine = EngineFromNetwork(
    NetworkFromOnnxPath(str(f"{onnx_in_dir}/clio4_mod.onnx"), strongly_typed=True),
    CreateConfig(
        profiles=profiles,
    ),
)

# Save the engine using polygraphy
print("!!! Saving TRT engine for block_in ... ")
with build_engine() as engine:
    save_engine(engine, path=str(trt_in_path))
print(f"Engine saved to {trt_in_path}")
print("========================================")
```

### Using the TRT Engine for Inference

We can use the `tensorrt` python package for running inference on the generated TRT engine. But keep in mind this is just a python wrapper on a very C/C++ runtime code. Hence a lot of overhead like managing memory and stuff falls on us to take care of.

So we can use this wrapper function below as reference and you can modify as required.

```python
from pathlib import Path
import logging
import tensorrt as trt
import torch
from cuda import cudart

from dit_train.tensorrt.plugins.fa3_plugin import CustomAttnPluginCreator


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
    def __init__(self, engine: trt.ICudaEngine, device: torch.device, profile_idx: int | None = None) -> None:
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
    def from_trt(cls, trt_file_path: Path, device: torch.device, profile_idx: int | None = None) -> "TRTEngine":
        runtime = trt.Runtime(trt.Logger())
        cudart.cudaSetDevice(device.index)

        plg_registry = trt.get_plugin_registry()
        my_plugin_creator = CustomAttnPluginCreator()
        plg_registry.register_creator(my_plugin_creator, "")

        # If a serialized engine exists, use it instead of building an engine.
        print (f"Reading TRT engine from file {trt_file_path} on device {device}")
        with open(trt_file_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
            if engine is None:
                raise RuntimeError(f"Failed to reload TRT cuda engine from {trt_file_path}.")
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
                    tensor = torch.empty(input_shape, dtype=TRT_DTYPE_TO_TORCH[dtype], device=self.device)
                    self.inputs[tensor_name] = tensor
                    self.inputs_bind[tensor_name] = i
                else:
                    assert input_shape is not None
                    output_shape = tuple(engine.get_tensor_shape(tensor_name))
                    # this assumes inputs and outputs have same dim and mapping of indices, check this for new engine
                    if -1 in output_shape:
                        index = output_shape.index(-1)
                        output_shape = list(output_shape)
                        output_shape[index] = input_shape[index]
                        output_shape = tuple(output_shape)
                    dtype = engine.get_tensor_dtype(tensor_name)
                    tensor = torch.empty(output_shape, dtype=TRT_DTYPE_TO_TORCH[dtype], device=self.device)
                    print (tensor_name, output_shape)
                    self.outputs[tensor_name] = tensor

                self.context.set_tensor_address(tensor_name, tensor.data_ptr())

    def execute(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        with torch.cuda.device(device=self.device):
            return self.execute_on_cuda(inputs)

    def execute_on_cuda(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        inputs_bind = self.inputs_bind
        for key, value in inputs.items():
            # self.inputs[key].copy_(value)
            self.inputs[key][:, : int(value.shape[1]), :].copy_(value)
            if key not in inputs_bind:
                raise KeyError(f"Invalid input name for TRT engine {key}.")

            self.context.set_input_shape(key, tuple(value.shape))
            self.context.set_tensor_address(key, value.data_ptr())

        self.context.execute_async_v3(stream_handle=self.stream)

        return self.outputs

def execute_trt(engine: TRTEngine, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    outputs = engine.execute_on_cuda(inputs)
    input_key = next(iter(inputs))
    input_seqlen = inputs[input_key].shape[1]

    for output_key in outputs: # we got multiple outputs
        outputs[output_key] = outputs[output_key][:, :input_seqlen, :]

    res = list(outputs.values())
    if len(res) > 1:
        return res
    else:
        return res[0]


class TrtEngineCallable(torch.nn.Module):
    def __init__(
        self,
        trt_engine: TRTEngine,
    ):
        super().__init__()
        self.trt_engine = trt_engine

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        input_dict = {
            "x": x,
        }
        return execute_trt(self.trt_engine, inputs=input_dict)


if __name__ == '__main__':
    trt_path = "/path/to/engine"
    trt_engine = TRTEngine.from_trt(trt_path, "cuda:0", 0) # last param is the opt profile index

    trt_fn = TrtEngineCallable(trt_engine)

    # Sample inference
    trt_fn(torch.randn(1,15255, 4096, dtype=torch.bfloat16, device="cuda:0"))
```

> Don't forget to add the code for registering the custom plugin before running the same.
> ```python
> plg_registry = trt.get_plugin_registry()
> my_plugin_creator = CustomAttnPluginCreator()
> plg_registry.register_creator(my_plugin_creator, "")
> ```
{: .prompt-info}
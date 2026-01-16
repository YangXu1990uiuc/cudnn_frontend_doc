# Quick Start Guide

This guide provides quick recipes for common operations. Each example is complete and ready to run.

## Setup Template

All examples use this setup:

```python
import cudnn
import torch

torch.manual_seed(42)
device = torch.device("cuda")
handle = cudnn.create_handle()
```

## Common Operations

### Convolution (2D Forward)

```python
# Input: [batch=4, channels=64, height=32, width=32]
x = torch.randn(4, 64, 32, 32, device=device, dtype=torch.float16).to(
    memory_format=torch.channels_last
)
# Weight: [out=128, in=64, kH=3, kW=3]
w = torch.randn(128, 64, 3, 3, device=device, dtype=torch.float16).to(
    memory_format=torch.channels_last
)

with cudnn.Graph(
    io_data_type=cudnn.data_type.HALF,
    compute_data_type=cudnn.data_type.FLOAT,
) as graph:
    y = graph.conv_fprop(image=x, weight=w, padding=[1, 1])
    y.set_output(True)

result = graph(x, w, handle=handle)
# Output shape: [4, 128, 32, 32]
```

### Matrix Multiplication

```python
# A: [batch=2, M=512, K=256]
A = torch.randn(2, 512, 256, device=device, dtype=torch.float16)
# B: [batch=2, K=256, N=1024]
B = torch.randn(2, 256, 1024, device=device, dtype=torch.float16)

with cudnn.Graph(
    io_data_type=cudnn.data_type.HALF,
    compute_data_type=cudnn.data_type.FLOAT,
) as graph:
    C = graph.matmul(A=A, B=B, name="gemm")
    C.set_output(True)

result = graph(A, B, handle=handle)
# Output shape: [2, 512, 1024]
```

### Scaled Dot-Product Attention (SDPA)

```python
import math

# Query, Key, Value tensors
batch, seq_len, num_heads, head_dim = 2, 1024, 8, 64
q = torch.randn(batch, num_heads, seq_len, head_dim, device=device, dtype=torch.float16)
k = torch.randn(batch, num_heads, seq_len, head_dim, device=device, dtype=torch.float16)
v = torch.randn(batch, num_heads, seq_len, head_dim, device=device, dtype=torch.float16)

with cudnn.Graph(
    io_data_type=cudnn.data_type.HALF,
    intermediate_data_type=cudnn.data_type.FLOAT,
    compute_data_type=cudnn.data_type.FLOAT,
) as graph:
    o, _ = graph.sdpa(
        q=q, k=k, v=v,
        attn_scale=1.0 / math.sqrt(head_dim),
        is_inference=True,
        use_causal_mask=True,  # For auto-regressive models
    )
    o.set_output(True).set_dim(q.shape).set_stride(q.stride())

result = graph(q, k, v, handle=handle)
# Output shape: [2, 8, 1024, 64]
```

### Layer Normalization

```python
# Input: [batch=4, seq_len=512, hidden=768]
x = torch.randn(4, 512, 768, device=device, dtype=torch.float16)
# Scale and Bias: [hidden=768]
scale = torch.ones(768, device=device, dtype=torch.float32)
bias = torch.zeros(768, device=device, dtype=torch.float32)
epsilon = 1e-5

with cudnn.Graph(
    io_data_type=cudnn.data_type.HALF,
    compute_data_type=cudnn.data_type.FLOAT,
) as graph:
    y, _, _ = graph.layernorm(
        input=x,
        scale=scale,
        bias=bias,
        epsilon=epsilon,
        name="ln"
    )
    y.set_output(True)

result = graph(x, scale, bias, handle=handle)
# Output shape: [4, 512, 768]
```

### Batch Normalization (Inference)

```python
# Input: [N=8, C=64, H=32, W=32]
x = torch.randn(8, 64, 32, 32, device=device, dtype=torch.float16).to(
    memory_format=torch.channels_last
)
# Statistics: [C=64]
mean = torch.zeros(64, device=device, dtype=torch.float32)
variance = torch.ones(64, device=device, dtype=torch.float32)
scale = torch.ones(64, device=device, dtype=torch.float32)
bias = torch.zeros(64, device=device, dtype=torch.float32)

with cudnn.Graph(
    io_data_type=cudnn.data_type.HALF,
    compute_data_type=cudnn.data_type.FLOAT,
) as graph:
    y = graph.batchnorm_inference(
        input=x,
        scale=scale,
        bias=bias,
        mean=mean,
        inv_variance=1.0 / torch.sqrt(variance + 1e-5),
    )
    y.set_output(True)

result = graph(x, scale, bias, mean, 1.0/torch.sqrt(variance + 1e-5), handle=handle)
```

## Advanced Patterns

### Fused Operations

Combine multiple operations for better performance:

```python
# Convolution + Bias + ReLU fused
x = torch.randn(4, 64, 32, 32, device=device, dtype=torch.float16).to(
    memory_format=torch.channels_last
)
w = torch.randn(128, 64, 3, 3, device=device, dtype=torch.float16).to(
    memory_format=torch.channels_last
)
b = torch.randn(1, 128, 1, 1, device=device, dtype=torch.float16).to(
    memory_format=torch.channels_last
)

with cudnn.Graph(
    io_data_type=cudnn.data_type.HALF,
    compute_data_type=cudnn.data_type.FLOAT,
) as graph:
    # Convolution
    conv_out = graph.conv_fprop(image=x, weight=w, padding=[1, 1])

    # Add bias
    bias_out = graph.bias(input=conv_out, bias=b)

    # ReLU activation
    relu_out = graph.relu(input=bias_out)
    relu_out.set_output(True)

result = graph(x, w, b, handle=handle)
```

### Graph Caching for Reuse

```python
# Build once, execute many times
with cudnn.Graph(
    io_data_type=cudnn.data_type.HALF,
    compute_data_type=cudnn.data_type.FLOAT,
) as graph:
    y = graph.conv_fprop(image=x, weight=w, padding=[1, 1])
    y.set_output(True)

# Execute multiple times with different data
for i in range(100):
    x_new = torch.randn_like(x)
    w_new = torch.randn_like(w)
    result = graph(x_new, w_new, handle=handle)
    # Graph is reused - no rebuild overhead!
```

### Using Dictionary Interface

For more flexibility in specifying inputs/outputs:

```python
with cudnn.Graph(
    io_data_type=cudnn.data_type.HALF,
    compute_data_type=cudnn.data_type.FLOAT,
) as graph:
    y = graph.conv_fprop(image=x, weight=w, padding=[1, 1], name="conv")
    y.set_output(True)

# Execute with dictionary
inputs = {
    "conv::image": x,
    "conv::weight": w,
}
outputs = graph(inputs, handle=handle)
result = outputs["conv::Y"]
```

## Performance Tips

!!! tip "Best Practices"

    1. **Use FP16/BF16** - Half precision is 2x faster on modern GPUs
    2. **Channels-last layout** - NHWC layout is optimized for tensor cores
    3. **Batch sizes** - Larger batches = better GPU utilization
    4. **Reuse graphs** - Build once, execute many times
    5. **Fuse operations** - Combine ops to reduce memory transfers

## Cleanup

Always clean up when done:

```python
cudnn.destroy_handle(handle)
```

## Next Steps

Now that you've seen common patterns, dive deeper into the concepts!

[Understanding Graphs :material-arrow-right:](../concepts/graphs.md){ .md-button .md-button--primary }

# LLM Quick Start

Most developers today are working with Large Language Models (LLMs). This quick start shows you how to use cuDNN Frontend for the key operations in LLMs - specifically **Scaled Dot-Product Attention (SDPA)**.

!!! tip "Why Start Here?"
    If you're building or optimizing LLMs like LLaMA, Mistral, GPT, etc., this is the most relevant starting point. cuDNN's SDPA implementation can give you 2-3x speedup over naive implementations.

## What You'll Build

We'll implement the **self-attention mechanism** - the core of every transformer:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

## Prerequisites

Before starting, make sure you have:

- NVIDIA GPU (Ampere A100 or newer recommended)
- Python 3.8+
- PyTorch with CUDA support
- cuDNN Frontend installed (`pip install nvidia-cudnn-frontend`)

## Step-by-Step Implementation

### Step 1: Setup

```python
import cudnn
import torch
import math

# Verify GPU and cuDNN
assert torch.cuda.is_available(), "Need CUDA GPU!"
assert torch.cuda.get_device_capability()[0] >= 8, "Need Ampere (SM80) or newer"

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"cuDNN version: {cudnn.backend_version()}")

# Create handle
handle = cudnn.create_handle()
```

### Step 2: Define LLM Dimensions

```python
# Typical LLM dimensions (similar to LLaMA 7B)
batch_size = 4
num_heads = 32
seq_len = 2048
head_dim = 128

# Scaling factor
attn_scale = 1.0 / math.sqrt(head_dim)

print(f"Configuration:")
print(f"  Batch size: {batch_size}")
print(f"  Sequence length: {seq_len}")
print(f"  Number of heads: {num_heads}")
print(f"  Head dimension: {head_dim}")
```

### Step 3: Create Q, K, V Tensors

```python
# In real LLMs, these come from linear projections of the input
# Shape: [batch, num_heads, seq_len, head_dim]

# Use bfloat16 - the standard for modern LLMs
dtype = torch.bfloat16

Q = torch.randn(batch_size, num_heads, seq_len, head_dim,
                device="cuda", dtype=dtype)
K = torch.randn(batch_size, num_heads, seq_len, head_dim,
                device="cuda", dtype=dtype)
V = torch.randn(batch_size, num_heads, seq_len, head_dim,
                device="cuda", dtype=dtype)

print(f"Q shape: {Q.shape}")  # [4, 32, 2048, 128]
```

### Step 4: Build the Attention Graph

```python
with cudnn.Graph(
    io_data_type=cudnn.data_type.BFLOAT16,       # Input/output in BF16
    intermediate_data_type=cudnn.data_type.FLOAT, # Softmax in FP32 for stability
    compute_data_type=cudnn.data_type.FLOAT,      # Accumulation in FP32
) as graph:

    # Scaled Dot-Product Attention
    output, stats = graph.sdpa(
        q=Q, k=K, v=V,
        attn_scale=attn_scale,
        is_inference=True,       # Set to False for training
        use_causal_mask=True,    # Autoregressive (decoder) attention
    )

    # Mark output - IMPORTANT: must set dimensions and strides
    output.set_output(True)
    output.set_dim(Q.shape)
    output.set_stride(Q.stride())

print("Graph built successfully!")
```

### Step 5: Execute

```python
# Run the attention
result = graph(Q, K, V, handle=handle)

print(f"Output shape: {result.shape}")  # [4, 32, 2048, 128]
print(f"Output dtype: {result.dtype}")  # torch.bfloat16
```

### Step 6: Verify Correctness

```python
# Compare with PyTorch's native SDPA
reference = torch.nn.functional.scaled_dot_product_attention(
    Q, K, V,
    scale=attn_scale,
    is_causal=True,
)

# Check results match
torch.testing.assert_close(result, reference, atol=1e-2, rtol=1e-2)
print("Results match PyTorch!")
```

### Step 7: Cleanup

```python
cudnn.destroy_handle(handle)
```

## Complete Code

```python
import cudnn
import torch
import math

# Setup
assert torch.cuda.is_available()
handle = cudnn.create_handle()

# LLM dimensions
batch_size, num_heads, seq_len, head_dim = 4, 32, 2048, 128
attn_scale = 1.0 / math.sqrt(head_dim)

# Create tensors
Q = torch.randn(batch_size, num_heads, seq_len, head_dim,
                device="cuda", dtype=torch.bfloat16)
K = torch.randn(batch_size, num_heads, seq_len, head_dim,
                device="cuda", dtype=torch.bfloat16)
V = torch.randn(batch_size, num_heads, seq_len, head_dim,
                device="cuda", dtype=torch.bfloat16)

# Build attention graph
with cudnn.Graph(
    io_data_type=cudnn.data_type.BFLOAT16,
    intermediate_data_type=cudnn.data_type.FLOAT,
    compute_data_type=cudnn.data_type.FLOAT,
) as graph:
    output, _ = graph.sdpa(
        q=Q, k=K, v=V,
        attn_scale=attn_scale,
        is_inference=True,
        use_causal_mask=True,
    )
    output.set_output(True).set_dim(Q.shape).set_stride(Q.stride())

# Execute
result = graph(Q, K, V, handle=handle)

# Verify
reference = torch.nn.functional.scaled_dot_product_attention(
    Q, K, V, scale=attn_scale, is_causal=True
)
torch.testing.assert_close(result, reference, atol=1e-2, rtol=1e-2)
print(f"Success! Attention output shape: {result.shape}")

# Cleanup
cudnn.destroy_handle(handle)
```

## Why cuDNN SDPA is Special

cuDNN implements **FlashAttention** algorithm internally:

| Feature | Standard Attention | cuDNN SDPA |
|---------|-------------------|------------|
| Memory | O(N²) | O(N) |
| Materializes attention matrix | Yes | No |
| Max sequence length | ~4K | 100K+ |
| Performance | Baseline | 2-3x faster |

## Grouped Query Attention (GQA)

Modern LLMs like LLaMA 2+ use GQA where K and V have fewer heads:

```python
# GQA: 32 Q heads, 8 KV heads
num_q_heads = 32
num_kv_heads = 8

Q = torch.randn(batch, num_q_heads, seq_len, head_dim, ...)
K = torch.randn(batch, num_kv_heads, seq_len, head_dim, ...)
V = torch.randn(batch, num_kv_heads, seq_len, head_dim, ...)

# cuDNN handles GQA automatically!
with cudnn.Graph(...) as graph:
    output, _ = graph.sdpa(q=Q, k=K, v=V, ...)
```

## Common Issues

!!! failure "Wrong tensor dimensions"
    Make sure Q, K, V are `[batch, heads, seq_len, head_dim]` not `[batch, seq_len, heads, head_dim]`. Use `.transpose(1, 2)` if needed.

!!! failure "Missing set_dim/set_stride on output"
    Always call both `set_dim()` and `set_stride()` on SDPA output tensors.

!!! failure "SM70 or older GPU"
    SDPA requires Ampere (SM80) or newer. RTX 30xx, A100, H100, RTX 40xx all work.

## What's Next?

Now that you understand attention, explore:

- [Full Transformer Layer](../api-reference/patterns.md) - Build complete LLM layers
- [Training with Backward Pass](../tutorials/attention.md) - Add gradient computation
- [Memory Optimization](../concepts/memory-performance.md) - Maximize throughput

[Explore Tutorials :material-arrow-right:](../tutorials/attention.md){ .md-button .md-button--primary }

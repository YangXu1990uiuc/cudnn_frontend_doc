# Python API Reference

This reference covers the **cuDNN Graph API** (Frontend v1) - the modern, recommended API for all new projects. The Graph API is designed for modern workloads like Large Language Models (LLMs) and Vision-Language Models (VLMs).

!!! info "Graph API vs Legacy API"
    This documentation focuses exclusively on the **Graph API** introduced in cuDNN Frontend v1.0. The legacy `pygraph` API (v0.x) is deprecated and should not be used for new projects.

## Core Classes

### cudnn.Graph

The primary interface for building computation graphs.

```python
class Graph:
    """
    Context manager for building cuDNN computation graphs.

    The Graph class provides a declarative way to define deep learning
    operations that are optimized and executed on NVIDIA GPUs.
    """

    def __init__(
        self,
        name: str = None,
        io_data_type: cudnn.data_type = None,
        intermediate_data_type: cudnn.data_type = None,
        compute_data_type: cudnn.data_type = None,
        handle: cudnn.Handle = None,
        heuristics: List[cudnn.heur_mode] = None,
        inputs: List[str] = None,
        outputs: List[str] = None,
    ):
        """
        Initialize a Graph.

        Args:
            name: Optional name for debugging
            io_data_type: Default data type for inputs/outputs
            intermediate_data_type: Data type for intermediate tensors
            compute_data_type: Data type for internal computations
            handle: cuDNN handle (can be provided at execution instead)
            heuristics: Algorithm selection modes (default: [A, FALLBACK])
            inputs: Input tensor names for positional execution
            outputs: Output tensor names for positional execution
        """
```

**Basic Usage:**

```python
# Context manager pattern (recommended)
with cudnn.Graph(
    io_data_type=cudnn.data_type.HALF,
    compute_data_type=cudnn.data_type.FLOAT,
) as graph:
    y = graph.conv_fprop(image=x, weight=w, padding=[1, 1])
    y.set_output(True)

result = graph(x, w, handle=handle)
```

**For LLM inference:**

```python
# Optimized for transformer inference
with cudnn.Graph(
    io_data_type=cudnn.data_type.HALF,
    intermediate_data_type=cudnn.data_type.FLOAT,  # Softmax precision
    compute_data_type=cudnn.data_type.FLOAT,
) as graph:
    # Self-attention
    output, _ = graph.sdpa(
        q=Q, k=K, v=V,
        attn_scale=1/math.sqrt(head_dim),
        is_inference=True,
        use_causal_mask=True,
    )
    output.set_output(True).set_dim(Q.shape).set_stride(Q.stride())
```

### cudnn.data_type

Enumeration of supported data types.

| Value | Description | Use Case |
|-------|-------------|----------|
| `FLOAT` | 32-bit floating point | Computation precision |
| `HALF` | 16-bit floating point (FP16) | LLM inference, training |
| `BFLOAT16` | Brain floating point 16 | Modern LLMs (LLaMA, etc.) |
| `FP8_E4M3` | 8-bit FP (4 exp, 3 mantissa) | High-throughput inference |
| `FP8_E5M2` | 8-bit FP (5 exp, 2 mantissa) | Gradient computation |
| `INT8` | 8-bit integer | Quantized inference |

**Recommended configurations for LLMs:**

```python
# Standard FP16 (most compatible)
io_data_type=cudnn.data_type.HALF
compute_data_type=cudnn.data_type.FLOAT

# BF16 for modern LLMs (LLaMA, Mistral, etc.)
io_data_type=cudnn.data_type.BFLOAT16
compute_data_type=cudnn.data_type.FLOAT

# FP8 for maximum throughput (Hopper GPUs)
io_data_type=cudnn.data_type.FP8_E4M3
compute_data_type=cudnn.data_type.FLOAT
```

### cudnn.heur_mode

Heuristic modes for algorithm selection.

| Value | Description |
|-------|-------------|
| `A` | Best heuristic (recommended primary) |
| `B` | Alternative heuristic |
| `FALLBACK` | Maximum compatibility |
| `CUDNN_FIND` | Exhaustive search (slow but optimal) |

## Graph Operations

### Attention Operations (LLM/VLM Core)

#### graph.sdpa

Scaled Dot-Product Attention - the heart of transformers.

```python
def sdpa(
    self,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attn_scale: float,
    is_inference: bool = True,
    use_causal_mask: bool = False,
    generate_stats: bool = False,
    attn_bias: Tensor = None,
    dropout: float = 0.0,
    dropout_seed: int = None,
    dropout_offset: int = 0,
    name: str = None,
) -> Tuple[Tensor, Tensor]:
    """
    Scaled Dot-Product Attention using FlashAttention algorithm.

    Computes: softmax(Q @ K.T / sqrt(d_k)) @ V

    Args:
        q: Query tensor [B, H, N, D] or [B, N, H, D] physical
        k: Key tensor [B, H, N, D] or [B, N, H, D] physical
        v: Value tensor [B, H, N, D] or [B, N, H, D] physical
        attn_scale: Scaling factor (usually 1/sqrt(head_dim))
        is_inference: True for inference, False for training
        use_causal_mask: Enable causal (autoregressive) masking
        generate_stats: Save stats for backward pass
        attn_bias: Optional attention bias [B, H, N, N]
        dropout: Dropout probability (training only)
        dropout_seed: Random seed for dropout
        dropout_offset: Offset for dropout RNG
        name: Operation name

    Returns:
        Tuple of (output, stats) tensors

    Supports:
        - Multi-Head Attention (MHA)
        - Grouped Query Attention (GQA)
        - Multi-Query Attention (MQA)
    """
```

**LLM Inference Example:**

```python
# Standard causal attention for LLM inference
with cudnn.Graph() as graph:
    output, _ = graph.sdpa(
        q=queries, k=keys, v=values,
        attn_scale=1.0 / math.sqrt(64),
        is_inference=True,
        use_causal_mask=True,  # Autoregressive decoding
    )
    output.set_output(True).set_dim(queries.shape).set_stride(queries.stride())
```

**GQA for LLaMA-style models:**

```python
# Grouped Query Attention (fewer KV heads than Q heads)
# Q: [B, 32, N, D], K/V: [B, 8, N, D]
with cudnn.Graph() as graph:
    output, _ = graph.sdpa(
        q=queries,  # 32 heads
        k=keys,     # 8 heads (4x fewer)
        v=values,   # 8 heads
        attn_scale=scale,
        is_inference=True,
        use_causal_mask=True,
    )
```

#### graph.sdpa_backward

Backward pass for SDPA (training).

```python
def sdpa_backward(
    self,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    o: Tensor,
    dO: Tensor,
    stats: Tensor,
    attn_scale: float,
    use_causal_mask: bool = False,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Backward pass for SDPA.

    Returns:
        Tuple of (dQ, dK, dV) gradients
    """
```

### Matrix Operations

#### graph.matmul

General matrix multiplication for linear layers.

```python
def matmul(
    self,
    A: Tensor,
    B: Tensor,
    compute_data_type: data_type = None,
    name: str = None,
) -> Tensor:
    """
    Matrix multiplication: C = A @ B

    Supports:
        - 2D: [M, K] @ [K, N] = [M, N]
        - Batched: [B, M, K] @ [B, K, N] = [B, M, N]
        - Mixed precision inputs

    Common LLM uses:
        - QKV projections
        - Output projections
        - FFN layers
    """
```

**LLM Linear Layers:**

```python
# FFN first layer (hidden expansion)
with cudnn.Graph() as graph:
    # x: [B, N, D], W1: [D, 4D]
    h = graph.matmul(x, W1)
    h = graph.gelu(h)
    # h: [B, N, 4D]

    # Second layer (hidden contraction)
    # W2: [4D, D]
    out = graph.matmul(h, W2)
    out.set_output(True)
```

### Normalization Operations

#### graph.layernorm

Layer normalization (used in all modern transformers).

```python
def layernorm(
    self,
    input: Tensor,
    scale: Tensor,
    bias: Tensor,
    epsilon: float = 1e-5,
    zero_centered_gamma: bool = False,
    name: str = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Layer Normalization: y = (x - mean) / sqrt(var + eps) * gamma + beta

    Args:
        input: Input tensor [B, N, D]
        scale: Scale parameter (gamma) [D]
        bias: Bias parameter (beta) [D]
        epsilon: Numerical stability constant
        zero_centered_gamma: Use (1 + gamma) instead of gamma

    Returns:
        Tuple of (output, mean, inv_variance)
    """
```

#### graph.rmsnorm

RMS Normalization (LLaMA, Mistral, etc.).

```python
def rmsnorm(
    self,
    input: Tensor,
    scale: Tensor,
    epsilon: float = 1e-5,
    name: str = None,
) -> Tuple[Tensor, Tensor]:
    """
    RMS Normalization: y = x / sqrt(mean(x^2) + eps) * gamma

    Simpler than LayerNorm - no mean centering.
    Used in LLaMA, Mistral, and other modern LLMs.

    Returns:
        Tuple of (output, inv_rms)
    """
```

**Pre-norm Transformer Pattern:**

```python
# Pre-norm (modern LLM style)
with cudnn.Graph() as graph:
    # RMS norm before attention
    x_norm, _ = graph.rmsnorm(x, gamma, epsilon=1e-5)
    # ... attention ...
    x = graph.add(x, attn_output)  # Residual

    # RMS norm before FFN
    x_norm, _ = graph.rmsnorm(x, gamma2, epsilon=1e-5)
    # ... FFN ...
    x = graph.add(x, ffn_output)  # Residual
```

### Activation Functions

#### graph.gelu

Gaussian Error Linear Unit (standard in transformers).

```python
def gelu(self, input: Tensor, name: str = None) -> Tensor:
    """GELU activation: x * Φ(x)"""
```

#### graph.silu / graph.swish

SiLU activation (used in SwiGLU).

```python
def silu(self, input: Tensor, name: str = None) -> Tensor:
    """SiLU/Swish activation: x * sigmoid(x)"""
```

#### graph.relu

ReLU activation.

```python
def relu(self, input: Tensor, name: str = None) -> Tensor:
    """ReLU activation: max(0, x)"""
```

### Element-wise Operations

```python
# Addition
y = graph.add(a=x1, b=x2)

# Multiplication
y = graph.mul(a=x1, b=x2)

# Bias addition
y = graph.bias(input=x, bias=b)

# Scaling
y = graph.scale(input=x, scale=s)
```

### Convolution Operations

For vision components in VLMs:

```python
# Forward convolution
y = graph.conv_fprop(
    image=x,           # [N, C, H, W]
    weight=w,          # [K, C, R, S]
    padding=[1, 1],
    stride=[1, 1],
    dilation=[1, 1],
)

# Data gradient (backward)
dx = graph.conv_dgrad(weight=w, loss=dy, padding=[1, 1])

# Weight gradient (backward)
dw = graph.conv_wgrad(image=x, loss=dy, padding=[1, 1])
```

## Tensor Methods

### set_output

Mark tensor as graph output.

```python
y.set_output(True)  # Required to retrieve this tensor
```

### set_name

Assign a name for referencing.

```python
y.set_name("attention_output")
```

### set_data_type

Override data type.

```python
y.set_data_type(cudnn.data_type.FLOAT)
```

### set_dim / set_stride

Set explicit dimensions and strides.

```python
y.set_dim([batch, num_heads, seq_len, head_dim])
y.set_stride([seq_len * num_heads * head_dim, head_dim, num_heads * head_dim, 1])
```

## Handle Management

```python
# Create handle
handle = cudnn.create_handle()

# Use in graph execution
result = graph(x, w, handle=handle)

# Destroy when done
cudnn.destroy_handle(handle)
```

## Graph Execution

### Positional Arguments

```python
with cudnn.Graph(
    inputs=["attn::q", "attn::k", "attn::v"],
    outputs=["output"],
) as graph:
    out, _ = graph.sdpa(q=Q, k=K, v=V, name="attn")
    out.set_output(True).set_name("output")

# Execute with positional args
result = graph(Q_data, K_data, V_data, handle=handle)
```

### Dictionary Interface

```python
with cudnn.Graph() as graph:
    out, _ = graph.sdpa(q=Q, k=K, v=V, name="attn")
    out.set_output(True)

# Execute with dict
result_dict = graph({
    "attn::q": Q_data,
    "attn::k": K_data,
    "attn::v": V_data,
}, handle=handle)
output = result_dict["attn::O"]
```

## Utility Functions

```python
# Get cuDNN backend version
version = cudnn.backend_version()  # e.g., 90300 for 9.3.0

# Query workspace size
workspace_size = graph.get_workspace_size()

# Serialize/deserialize graphs
serialized = graph.serialize()
graph = cudnn.Graph.deserialize(serialized)
```

## Complete LLM Layer Example

```python
import cudnn
import torch
import math

def create_llm_attention_layer(
    batch_size, seq_len, num_heads, head_dim, handle
):
    """Creates an optimized attention layer for LLM inference."""

    # Create template tensors
    hidden_dim = num_heads * head_dim
    x = torch.randn(batch_size, seq_len, hidden_dim,
                    device="cuda", dtype=torch.bfloat16)

    # RMSNorm parameters
    gamma = torch.ones(hidden_dim, device="cuda", dtype=torch.float32)

    # Projection weights (fused QKV)
    W_qkv = torch.randn(hidden_dim, 3 * hidden_dim,
                        device="cuda", dtype=torch.bfloat16)
    W_out = torch.randn(hidden_dim, hidden_dim,
                        device="cuda", dtype=torch.bfloat16)

    # Build the graph
    with cudnn.Graph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    ) as graph:
        # Pre-attention RMSNorm
        x_norm, _ = graph.rmsnorm(x, gamma, epsilon=1e-5)

        # QKV projection would go here (using matmul)
        # For this example, we assume Q, K, V are pre-computed

    return graph
```

## Next Steps

See the C++ API reference for equivalent C++ interfaces.

[C++ API Reference :material-arrow-right:](cpp.md){ .md-button .md-button--primary }

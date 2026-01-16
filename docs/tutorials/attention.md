# Attention (SDPA) Tutorial

Scaled Dot-Product Attention (SDPA) is the heart of transformer models. cuDNN Frontend provides highly optimized implementations based on FlashAttention.

## What is SDPA?

SDPA computes attention in transformers:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

```mermaid
graph LR
    subgraph "Inputs"
        Q[Query<br>B×H×N×D]
        K[Key<br>B×H×N×D]
        V[Value<br>B×H×N×D]
    end

    subgraph "Attention"
        MM1[Q @ K.T]
        Scale[÷ √d]
        Soft[Softmax]
        MM2[@ V]
    end

    subgraph "Output"
        O[Output<br>B×H×N×D]
    end

    Q --> MM1
    K --> MM1
    MM1 --> Scale --> Soft --> MM2
    V --> MM2
    MM2 --> O
```

## Why cuDNN SDPA?

Standard attention has O(N²) memory complexity. cuDNN's implementation uses **FlashAttention** algorithm:

| Aspect | Standard Attention | cuDNN SDPA (FlashAttention) |
|--------|-------------------|----------------------------|
| Memory | O(N²) | O(N) |
| Speed | Limited by memory | Compute-bound |
| Sequence length | ~4K max | 128K+ possible |

## Basic SDPA

```python
import cudnn
import torch
import math

torch.manual_seed(42)
device = torch.device("cuda")
handle = cudnn.create_handle()

# Typical transformer dimensions
batch_size = 8
num_heads = 12
seq_len = 1024
head_dim = 64

# Create Q, K, V tensors
# Shape: [batch, num_heads, seq_len, head_dim]
Q = torch.randn(batch_size, num_heads, seq_len, head_dim,
                device=device, dtype=torch.float16)
K = torch.randn(batch_size, num_heads, seq_len, head_dim,
                device=device, dtype=torch.float16)
V = torch.randn(batch_size, num_heads, seq_len, head_dim,
                device=device, dtype=torch.float16)

# Scaling factor
attn_scale = 1.0 / math.sqrt(head_dim)

# Build SDPA graph
with cudnn.Graph(
    io_data_type=cudnn.data_type.HALF,
    intermediate_data_type=cudnn.data_type.FLOAT,  # Softmax in FP32
    compute_data_type=cudnn.data_type.FLOAT,
) as graph:
    output, stats = graph.sdpa(
        q=Q,
        k=K,
        v=V,
        attn_scale=attn_scale,
        is_inference=True,
    )
    output.set_output(True).set_dim(Q.shape).set_stride(Q.stride())

# Execute
result = graph(Q, K, V, handle=handle)
print(f"Output shape: {result.shape}")  # [8, 12, 1024, 64]

# Verify with PyTorch
reference = torch.nn.functional.scaled_dot_product_attention(
    Q, K, V, scale=attn_scale
)
torch.testing.assert_close(result, reference, atol=5e-3, rtol=3e-3)
```

## Causal Attention (Autoregressive)

For language models, use causal masking:

```python
with cudnn.Graph(
    io_data_type=cudnn.data_type.HALF,
    intermediate_data_type=cudnn.data_type.FLOAT,
    compute_data_type=cudnn.data_type.FLOAT,
) as graph:
    output, stats = graph.sdpa(
        q=Q, k=K, v=V,
        attn_scale=attn_scale,
        is_inference=True,
        use_causal_mask=True,  # Enable causal masking!
    )
    output.set_output(True).set_dim(Q.shape).set_stride(Q.stride())

result = graph(Q, K, V, handle=handle)

# Verify with PyTorch causal attention
reference = torch.nn.functional.scaled_dot_product_attention(
    Q, K, V, scale=attn_scale, is_causal=True
)
torch.testing.assert_close(result, reference, atol=5e-3, rtol=3e-3)
```

## Training with Backward Pass

For training, you need stats from forward and backward gradients:

```python
# Forward pass (training mode)
with cudnn.Graph(
    io_data_type=cudnn.data_type.HALF,
    intermediate_data_type=cudnn.data_type.FLOAT,
    compute_data_type=cudnn.data_type.FLOAT,
) as fwd_graph:
    output, stats = fwd_graph.sdpa(
        q=Q, k=K, v=V,
        attn_scale=attn_scale,
        is_inference=False,      # Training mode
        generate_stats=True,     # Need stats for backward
        use_causal_mask=True,
    )
    output.set_output(True).set_dim(Q.shape).set_stride(Q.stride())
    stats.set_output(True)  # Save stats for backward

# Execute forward
O, S = fwd_graph(Q, K, V, handle=handle)

# Gradient from downstream
dO = torch.randn_like(O)

# Backward pass
with cudnn.Graph(
    io_data_type=cudnn.data_type.HALF,
    intermediate_data_type=cudnn.data_type.FLOAT,
    compute_data_type=cudnn.data_type.FLOAT,
) as bwd_graph:
    dQ, dK, dV = bwd_graph.sdpa_backward(
        q=Q, k=K, v=V, o=O,
        dO=dO,
        stats=S,
        attn_scale=attn_scale,
        use_causal_mask=True,
    )
    dQ.set_output(True).set_dim(Q.shape).set_stride(Q.stride())
    dK.set_output(True).set_dim(K.shape).set_stride(K.stride())
    dV.set_output(True).set_dim(V.shape).set_stride(V.stride())

# Execute backward
dQ_out, dK_out, dV_out = bwd_graph(Q, K, V, O, dO, S, handle=handle)
```

## Different Tensor Layouts

cuDNN supports different memory layouts:

### BHSD (Batch, Head, Seq, Dim) - Default

```python
# Standard layout
Q = torch.randn(batch, num_heads, seq_len, head_dim, device=device)
# Strides: (num_heads*seq_len*head_dim, seq_len*head_dim, head_dim, 1)
```

### BSHD (Batch, Seq, Head, Dim) - Interleaved

More efficient for some operations:

```python
# Create in BSHD then view as BHSD
Q = torch.randn(batch, seq_len, num_heads, head_dim, device=device)
Q = Q.transpose(1, 2)  # Logical BHSD, physical BSHD
# This is the recommended layout!
```

## Grouped Query Attention (GQA)

Used in LLaMA 2+, Mistral, etc.:

```python
# GQA: fewer K, V heads than Q heads
num_q_heads = 32
num_kv_heads = 8  # K, V have fewer heads
head_dim = 128

Q = torch.randn(batch, num_q_heads, seq_len, head_dim, device=device, dtype=torch.float16)
K = torch.randn(batch, num_kv_heads, seq_len, head_dim, device=device, dtype=torch.float16)
V = torch.randn(batch, num_kv_heads, seq_len, head_dim, device=device, dtype=torch.float16)

# cuDNN handles GQA automatically when head counts differ
with cudnn.Graph() as graph:
    output, _ = graph.sdpa(
        q=Q, k=K, v=V,
        attn_scale=1.0/math.sqrt(head_dim),
        is_inference=True,
    )
    output.set_output(True).set_dim(Q.shape).set_stride(Q.stride())
```

## Multi-Query Attention (MQA)

Extreme case: single K, V head:

```python
# MQA: 1 K, V head shared across all Q heads
Q = torch.randn(batch, 32, seq_len, head_dim, device=device, dtype=torch.float16)
K = torch.randn(batch, 1, seq_len, head_dim, device=device, dtype=torch.float16)
V = torch.randn(batch, 1, seq_len, head_dim, device=device, dtype=torch.float16)

with cudnn.Graph() as graph:
    output, _ = graph.sdpa(q=Q, k=K, v=V, attn_scale=attn_scale)
    output.set_output(True).set_dim(Q.shape).set_stride(Q.stride())
```

## Paged KV Cache (Inference)

For efficient LLM serving with PagedAttention:

```python
# For inference with non-contiguous KV cache
# See: samples/python/52_sdpa_with_paged_caches.ipynb

# Page table maps virtual blocks to physical blocks
page_size = 256
num_pages = 64

# K cache stored in pages
K_cache = torch.randn(num_pages, page_size, num_heads, head_dim,
                       device=device, dtype=torch.float16)

# Page table: [batch, num_blocks_per_seq]
page_table = torch.randint(0, num_pages, (batch, seq_len // page_size),
                            device=device, dtype=torch.int32)

# cuDNN handles paged attention
with cudnn.Graph() as graph:
    output, _ = graph.sdpa_with_paged_cache(
        q=Q, k=K_cache, v=V_cache,
        page_table=page_table,
        # ... other params
    )
```

## Dropout in Attention

For training regularization:

```python
# With dropout
dropout_prob = 0.1
seed = 12345

with cudnn.Graph() as graph:
    output, stats = graph.sdpa(
        q=Q, k=K, v=V,
        attn_scale=attn_scale,
        is_inference=False,
        dropout=dropout_prob,
        dropout_seed=seed,
        dropout_offset=0,
    )
    output.set_output(True).set_dim(Q.shape).set_stride(Q.stride())
```

## Attention with Bias

For relative position encoding:

```python
# Attention bias: [batch, num_heads, seq_len, seq_len]
# Or: [1, 1, seq_len, seq_len] for shared bias
attn_bias = torch.randn(1, 1, seq_len, seq_len, device=device, dtype=torch.float16)

with cudnn.Graph() as graph:
    output, _ = graph.sdpa(
        q=Q, k=K, v=V,
        attn_scale=attn_scale,
        attn_bias=attn_bias,  # Added to Q@K before softmax
        is_inference=True,
    )
    output.set_output(True).set_dim(Q.shape).set_stride(Q.stride())
```

## ALiBi (Attention with Linear Biases)

Popular in some models:

```python
# Create ALiBi slopes
def get_alibi_slopes(num_heads):
    closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
    base = 2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3)))
    powers = torch.arange(1, closest_power_of_2 + 1)
    slopes = base ** powers
    if closest_power_of_2 != num_heads:
        extra_powers = torch.arange(1, 2 * (num_heads - closest_power_of_2) + 1, 2)
        extra_slopes = (base ** 0.5) ** extra_powers
        slopes = torch.cat([slopes, extra_slopes])
    return slopes

slopes = get_alibi_slopes(num_heads).to(device)

# Create position-based bias
positions = torch.arange(seq_len, device=device)
alibi = slopes[:, None, None] * (positions[None, :, None] - positions[None, None, :])
# alibi shape: [num_heads, seq_len, seq_len]
```

## Performance Tips

!!! tip "SDPA Optimization"

    1. **Use BSHD layout**: Most efficient for tensor cores
    2. **Align sequence lengths**: Multiples of 64 or 128
    3. **Prefer FP16/BF16**: Significant speedup
    4. **Use causal mask**: Built-in optimization vs explicit mask
    5. **Batch queries**: Larger batches = better efficiency

## Comparison with PyTorch

```python
# cuDNN SDPA
with cudnn.Graph() as graph:
    output, _ = graph.sdpa(q=Q, k=K, v=V, attn_scale=attn_scale, is_inference=True)
    output.set_output(True).set_dim(Q.shape).set_stride(Q.stride())
cudnn_out = graph(Q, K, V, handle=handle)

# PyTorch SDPA (also uses FlashAttention when available)
torch_out = torch.nn.functional.scaled_dot_product_attention(
    Q, K, V, scale=attn_scale
)

# Verify
torch.testing.assert_close(cudnn_out, torch_out, atol=5e-3, rtol=3e-3)
```

## Complete Attention Layer Example

```python
class CudnnAttention:
    def __init__(self, hidden_dim, num_heads, handle):
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.handle = handle
        self.attn_scale = 1.0 / math.sqrt(self.head_dim)

        # Projection weights (would normally be nn.Parameter)
        self.W_qkv = torch.randn(hidden_dim, 3 * hidden_dim,
                                  device="cuda", dtype=torch.float16)
        self.W_out = torch.randn(hidden_dim, hidden_dim,
                                  device="cuda", dtype=torch.float16)

    def forward(self, x):
        batch, seq_len, _ = x.shape

        # QKV projection (could also use cuDNN matmul)
        qkv = x @ self.W_qkv
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape for attention
        q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # cuDNN SDPA
        with cudnn.Graph() as graph:
            out, _ = graph.sdpa(
                q=q, k=k, v=v,
                attn_scale=self.attn_scale,
                is_inference=True,
                use_causal_mask=True,
            )
            out.set_output(True).set_dim(q.shape).set_stride(q.stride())

        attn_out = graph(q, k, v, handle=self.handle)

        # Reshape and output projection
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return attn_out @ self.W_out
```

## Cleanup

```python
cudnn.destroy_handle(handle)
```

## Next Steps

Learn about normalization operations.

[Normalization Tutorial :material-arrow-right:](normalization.md){ .md-button .md-button--primary }

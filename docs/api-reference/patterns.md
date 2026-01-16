# Common Patterns

This page provides production-ready patterns for common deep learning operations, focused on LLM and VLM workloads.

!!! warning "What cuDNN Can Actually Fuse"
    cuDNN graphs have limits on what can be fused together. A single graph typically handles:

    - **SDPA**: Q, K, V → attention output (this IS heavily fused internally)
    - **Normalization**: input → normalized output (with optional bias/scale)
    - **Conv + activation**: convolution → activation (e.g., ReLU, SiLU)
    - **MatMul + bias + activation**: limited fusion

    A full transformer layer requires **multiple separate graphs** plus PyTorch operations. The examples below show realistic patterns, not magical full-layer fusion.

## LLM Patterns

### Complete Transformer Layer

The standard pre-norm transformer block - note this uses **multiple small graphs**, not one fused graph:

```python
import cudnn
import torch
import math

class TransformerLayer:
    """
    Pre-norm transformer layer as used in LLaMA, Mistral, etc.

    Architecture:
    x → RMSNorm → Self-Attention → (+x) → RMSNorm → FFN → (+x) → y
    """

    def __init__(self, hidden_dim, num_heads, ff_dim, handle):
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.ff_dim = ff_dim
        self.handle = handle
        self.attn_scale = 1.0 / math.sqrt(self.head_dim)

        # Initialize parameters (in production, these would be loaded)
        self._init_parameters()

    def _init_parameters(self):
        device = "cuda"
        dtype = torch.bfloat16

        # RMSNorm parameters
        self.norm1_weight = torch.ones(self.hidden_dim, device=device, dtype=torch.float32)
        self.norm2_weight = torch.ones(self.hidden_dim, device=device, dtype=torch.float32)

        # Attention projections (fused QKV)
        self.W_qkv = torch.randn(self.hidden_dim, 3 * self.hidden_dim,
                                  device=device, dtype=dtype) * 0.02
        self.W_out = torch.randn(self.hidden_dim, self.hidden_dim,
                                  device=device, dtype=dtype) * 0.02

        # FFN (SwiGLU)
        self.W_gate = torch.randn(self.hidden_dim, self.ff_dim,
                                   device=device, dtype=dtype) * 0.02
        self.W_up = torch.randn(self.hidden_dim, self.ff_dim,
                                 device=device, dtype=dtype) * 0.02
        self.W_down = torch.randn(self.ff_dim, self.hidden_dim,
                                   device=device, dtype=dtype) * 0.02

    def forward(self, x, kv_cache=None):
        batch, seq_len, _ = x.shape

        # === Self-Attention ===
        residual = x

        # Pre-attention RMSNorm
        with cudnn.Graph(io_data_type=cudnn.data_type.BFLOAT16,
                         compute_data_type=cudnn.data_type.FLOAT) as norm_graph:
            x_norm, _ = norm_graph.rmsnorm(x, self.norm1_weight, epsilon=1e-5)
            x_norm.set_output(True)
        x_norm = norm_graph(x, self.norm1_weight, handle=self.handle)

        # QKV projection
        qkv = x_norm @ self.W_qkv
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape for attention
        q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # cuDNN SDPA
        with cudnn.Graph(io_data_type=cudnn.data_type.BFLOAT16,
                         intermediate_data_type=cudnn.data_type.FLOAT,
                         compute_data_type=cudnn.data_type.FLOAT) as attn_graph:
            attn_out, _ = attn_graph.sdpa(
                q=q, k=k, v=v,
                attn_scale=self.attn_scale,
                is_inference=True,
                use_causal_mask=True,
            )
            attn_out.set_output(True).set_dim(q.shape).set_stride(q.stride())

        attn_out = attn_graph(q, k, v, handle=self.handle)

        # Output projection
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        attn_out = attn_out @ self.W_out

        # Residual
        x = residual + attn_out

        # === FFN (SwiGLU) ===
        residual = x

        # Pre-FFN RMSNorm
        with cudnn.Graph(io_data_type=cudnn.data_type.BFLOAT16,
                         compute_data_type=cudnn.data_type.FLOAT) as norm2_graph:
            x_norm, _ = norm2_graph.rmsnorm(x, self.norm2_weight, epsilon=1e-5)
            x_norm.set_output(True)
        x_norm = norm2_graph(x, self.norm2_weight, handle=self.handle)

        # SwiGLU: (x @ W_up) * silu(x @ W_gate) @ W_down
        gate = x_norm @ self.W_gate
        up = x_norm @ self.W_up

        with cudnn.Graph(io_data_type=cudnn.data_type.BFLOAT16,
                         compute_data_type=cudnn.data_type.FLOAT) as swiglu_graph:
            gate_activated = swiglu_graph.silu(gate)
            gated = swiglu_graph.mul(up, gate_activated)
            gated.set_output(True)
        gated = swiglu_graph(gate, up, handle=self.handle)

        ff_out = gated @ self.W_down

        # Residual
        x = residual + ff_out

        return x
```

### Grouped Query Attention (GQA)

Used in LLaMA 2+, Mistral:

```python
def gqa_attention(q, k, v, num_kv_groups, handle):
    """
    Grouped Query Attention where K, V have fewer heads than Q.

    Args:
        q: [batch, num_q_heads, seq_len, head_dim]
        k: [batch, num_kv_heads, seq_len, head_dim]
        v: [batch, num_kv_heads, seq_len, head_dim]
        num_kv_groups: num_q_heads // num_kv_heads
    """
    batch, num_q_heads, seq_len, head_dim = q.shape
    num_kv_heads = k.shape[1]

    # cuDNN handles GQA natively when head counts differ
    with cudnn.Graph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    ) as graph:
        output, _ = graph.sdpa(
            q=q, k=k, v=v,
            attn_scale=1.0 / math.sqrt(head_dim),
            is_inference=True,
            use_causal_mask=True,
        )
        output.set_output(True).set_dim(q.shape).set_stride(q.stride())

    return graph(q, k, v, handle=handle)
```

### Rotary Position Embedding (RoPE)

Used in most modern LLMs:

```python
def apply_rotary_embedding(q, k, cos, sin):
    """
    Apply RoPE to query and key tensors.

    This is typically done before SDPA.
    """
    # Split into pairs
    q1, q2 = q[..., ::2], q[..., 1::2]
    k1, k2 = k[..., ::2], k[..., 1::2]

    # Apply rotation
    q_rotated = torch.cat([
        q1 * cos - q2 * sin,
        q1 * sin + q2 * cos
    ], dim=-1)

    k_rotated = torch.cat([
        k1 * cos - k2 * sin,
        k1 * sin + k2 * cos
    ], dim=-1)

    return q_rotated, k_rotated

# Usage with cuDNN SDPA
q_rope, k_rope = apply_rotary_embedding(q, k, cos, sin)
# Then use q_rope, k_rope in SDPA
```

## VLM Patterns

### Vision Encoder (ViT-style)

```python
class VisionEncoder:
    """Vision encoder for VLMs like LLaVA, Qwen-VL."""

    def __init__(self, image_size, patch_size, hidden_dim, num_heads, num_layers, handle):
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.hidden_dim = hidden_dim
        self.handle = handle

        # Patch embedding (convolution)
        self.patch_embed = torch.randn(
            hidden_dim, 3, patch_size, patch_size,
            device="cuda", dtype=torch.float16
        ).to(memory_format=torch.channels_last)

        # Position embedding
        self.pos_embed = torch.randn(
            1, self.num_patches + 1, hidden_dim,
            device="cuda", dtype=torch.float16
        )

        # CLS token
        self.cls_token = torch.randn(
            1, 1, hidden_dim,
            device="cuda", dtype=torch.float16
        )

    def patch_embedding(self, images):
        """Convert images to patch embeddings using convolution."""
        batch = images.shape[0]

        with cudnn.Graph(
            io_data_type=cudnn.data_type.HALF,
            compute_data_type=cudnn.data_type.FLOAT,
        ) as graph:
            patches = graph.conv_fprop(
                images, self.patch_embed,
                padding=[0, 0],
                stride=[self.patch_size, self.patch_size],
            )
            patches.set_output(True)

        patches = graph(images, self.patch_embed, handle=self.handle)
        # Reshape: [B, D, H/P, W/P] -> [B, num_patches, D]
        patches = patches.flatten(2).transpose(1, 2)
        return patches

    def forward(self, images):
        # Patch embedding
        x = self.patch_embedding(images)

        # Add CLS token
        cls_tokens = self.cls_token.expand(images.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add position embedding
        x = x + self.pos_embed

        # Apply transformer layers (similar to LLM layers above)
        # ...

        return x
```

### Cross-Attention for VLM

```python
def cross_attention(query, key, value, handle):
    """
    Cross-attention between text and vision features.

    Args:
        query: Text features [batch, text_len, hidden_dim]
        key: Vision features [batch, vision_len, hidden_dim]
        value: Vision features [batch, vision_len, hidden_dim]
    """
    batch, text_len, hidden_dim = query.shape
    vision_len = key.shape[1]

    # Reshape for multi-head attention
    num_heads = 12
    head_dim = hidden_dim // num_heads

    q = query.view(batch, text_len, num_heads, head_dim).transpose(1, 2)
    k = key.view(batch, vision_len, num_heads, head_dim).transpose(1, 2)
    v = value.view(batch, vision_len, num_heads, head_dim).transpose(1, 2)

    # Cross-attention (no causal mask!)
    with cudnn.Graph(
        io_data_type=cudnn.data_type.HALF,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    ) as graph:
        output, _ = graph.sdpa(
            q=q, k=k, v=v,
            attn_scale=1.0 / math.sqrt(head_dim),
            is_inference=True,
            use_causal_mask=False,  # No causal mask for cross-attention!
        )
        output.set_output(True).set_dim(q.shape).set_stride(q.stride())

    output = graph(q, k, v, handle=handle)
    return output.transpose(1, 2).contiguous().view(batch, text_len, hidden_dim)
```

## Production Patterns

### Graph Caching for Inference Server

```python
from functools import lru_cache
from typing import Tuple

class GraphCache:
    """Cache cuDNN graphs for different configurations."""

    def __init__(self, handle):
        self.handle = handle
        self._cache = {}

    def get_attention_graph(
        self,
        batch_size: int,
        num_heads: int,
        seq_len: int,
        head_dim: int,
    ) -> cudnn.Graph:
        """Get or create attention graph for given dimensions."""
        key = ("attention", batch_size, num_heads, seq_len, head_dim)

        if key not in self._cache:
            self._cache[key] = self._build_attention_graph(
                batch_size, num_heads, seq_len, head_dim
            )

        return self._cache[key]

    def _build_attention_graph(self, batch, heads, seq, dim):
        q = torch.randn(batch, heads, seq, dim, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(batch, heads, seq, dim, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(batch, heads, seq, dim, device="cuda", dtype=torch.bfloat16)

        with cudnn.Graph(
            io_data_type=cudnn.data_type.BFLOAT16,
            intermediate_data_type=cudnn.data_type.FLOAT,
            compute_data_type=cudnn.data_type.FLOAT,
        ) as graph:
            o, _ = graph.sdpa(q, k, v, attn_scale=1/math.sqrt(dim),
                              is_inference=True, use_causal_mask=True)
            o.set_output(True).set_dim(q.shape).set_stride(q.stride())

        return graph

# Usage in inference server
cache = GraphCache(handle)

def process_request(batch_size, seq_len):
    graph = cache.get_attention_graph(batch_size, 32, seq_len, 128)
    return graph(q, k, v, handle=handle)
```

### Batched Inference with Dynamic Shapes

```python
class DynamicBatchProcessor:
    """Handle variable batch sizes efficiently."""

    def __init__(self, max_batch_size, num_heads, max_seq_len, head_dim, handle):
        self.handle = handle
        self.graphs = {}

        # Pre-build graphs for common batch sizes
        for batch_size in [1, 2, 4, 8, 16, 32]:
            if batch_size <= max_batch_size:
                self.graphs[batch_size] = self._build_graph(
                    batch_size, num_heads, max_seq_len, head_dim
                )

    def _build_graph(self, batch, heads, seq, dim):
        # Build graph for specific batch size
        ...

    def process(self, q, k, v):
        batch_size = q.shape[0]

        # Find closest pre-built graph
        valid_sizes = [s for s in self.graphs.keys() if s >= batch_size]
        if not valid_sizes:
            raise ValueError(f"Batch size {batch_size} too large")

        graph = self.graphs[min(valid_sizes)]

        # Pad if necessary
        if batch_size < min(valid_sizes):
            # Pad tensors to match graph dimensions
            pad_size = min(valid_sizes) - batch_size
            q = torch.cat([q, torch.zeros(pad_size, *q.shape[1:], device=q.device)], dim=0)
            k = torch.cat([k, torch.zeros(pad_size, *k.shape[1:], device=k.device)], dim=0)
            v = torch.cat([v, torch.zeros(pad_size, *v.shape[1:], device=v.device)], dim=0)

        output = graph(q, k, v, handle=self.handle)

        # Remove padding
        return output[:batch_size]
```

### FP8 Inference (Hopper+)

```python
def fp8_attention(q_fp8, k_fp8, v_fp8, q_scale, k_scale, v_scale, handle):
    """
    FP8 attention for maximum throughput on Hopper GPUs.

    Requires proper quantization of inputs.
    """
    with cudnn.Graph(
        io_data_type=cudnn.data_type.FP8_E4M3,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    ) as graph:
        output, _ = graph.sdpa(
            q=q_fp8, k=k_fp8, v=v_fp8,
            attn_scale=1.0 / math.sqrt(128),  # Includes scaling
            is_inference=True,
            use_causal_mask=True,
            # FP8 specific parameters
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
        )
        output.set_output(True)
        output.set_data_type(cudnn.data_type.BFLOAT16)  # Output in BF16

    return graph(q_fp8, k_fp8, v_fp8, handle=handle)
```

## Error Handling Pattern

```python
def safe_execute(graph, *args, handle, fallback_fn=None):
    """Execute graph with fallback on error."""
    try:
        return graph(*args, handle=handle)
    except Exception as e:
        print(f"cuDNN execution failed: {e}")
        if fallback_fn:
            print("Using fallback implementation")
            return fallback_fn(*args)
        raise

# Usage
result = safe_execute(
    attention_graph,
    q, k, v,
    handle=handle,
    fallback_fn=lambda q, k, v: torch.nn.functional.scaled_dot_product_attention(q, k, v)
)
```

## Performance Monitoring

```python
import time

class PerformanceMonitor:
    """Monitor cuDNN graph execution performance."""

    def __init__(self):
        self.timings = {}

    def time_execution(self, name, graph, *args, handle, warmup=3, iterations=10):
        # Warmup
        for _ in range(warmup):
            graph(*args, handle=handle)
        torch.cuda.synchronize()

        # Timed runs
        start = time.perf_counter()
        for _ in range(iterations):
            graph(*args, handle=handle)
        torch.cuda.synchronize()
        end = time.perf_counter()

        avg_time = (end - start) / iterations * 1000  # ms
        self.timings[name] = avg_time
        return avg_time

# Usage
monitor = PerformanceMonitor()
attn_time = monitor.time_execution("attention", attn_graph, q, k, v, handle=handle)
print(f"Attention: {attn_time:.2f} ms")
```

## Summary

| Pattern | Use Case | Key Points |
|---------|----------|------------|
| Pre-norm Transformer | LLM layers | RMSNorm → Attention → Residual |
| GQA | Modern LLMs | Fewer KV heads than Q heads |
| RoPE | Position encoding | Apply before SDPA |
| Vision Encoder | VLMs | Conv → Flatten → Transformer |
| Cross-Attention | VLMs | No causal mask |
| Graph Caching | Inference servers | Build once, execute many |
| FP8 | Maximum throughput | Hopper GPUs only |

# Frequently Asked Questions

Common questions and answers about cuDNN Frontend.

## General Questions

### What is the difference between cuDNN and cuDNN Frontend?

**cuDNN** is NVIDIA's core library with highly optimized GPU kernels for deep learning operations (convolutions, attention, normalization, etc.).

**cuDNN Frontend** is a modern API layer that makes cuDNN easier to use:

| Aspect | cuDNN (Backend) | cuDNN Frontend |
|--------|----------------|----------------|
| API style | Descriptor-based C API | Graph-based Python/C++ |
| Code required | Hundreds of lines | Tens of lines |
| Optimization | Manual | Automatic |
| Learning curve | Steep | Gentle |

### When should I use cuDNN Frontend directly vs PyTorch?

| Scenario | Recommendation |
|----------|----------------|
| Standard models (ResNet, BERT) | Use PyTorch - it uses cuDNN internally |
| Custom fused operations | Use cuDNN Frontend directly |
| Maximum performance tuning | Use cuDNN Frontend with autotuning |
| Production inference systems | Consider cuDNN Frontend |
| Exotic architectures | Use cuDNN Frontend |

### What GPU architectures are supported?

| Architecture | GPUs | SDPA Support |
|--------------|------|--------------|
| Volta (SM70) | V100 | No |
| Turing (SM75) | RTX 20xx, T4 | No |
| Ampere (SM80) | A100, RTX 30xx | Yes |
| Ada (SM89) | RTX 40xx | Yes |
| Hopper (SM90) | H100 | Yes (+ FP8) |
| Blackwell (SM100) | B100, RTX 50xx | Yes |

SDPA (Scaled Dot-Product Attention) requires Ampere or newer.

## Installation Issues

### "CUDA not available" error

```python
>>> import torch
>>> torch.cuda.is_available()
False
```

**Solutions:**

1. Check if GPU is recognized: `nvidia-smi`
2. Install CUDA-enabled PyTorch:
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu121
   ```
3. Verify CUDA toolkit: `nvcc --version`

### "cuDNN version mismatch" error

**Solution:** Install matching versions:

```bash
pip install nvidia-cudnn-cu12==9.0.0
pip install nvidia-cudnn-frontend
```

### Import error: "No module named 'cudnn'"

**Solution:**

```bash
pip uninstall nvidia-cudnn-frontend
pip install nvidia-cudnn-frontend
```

## Graph Building Issues

### "No execution plan found" error

This means cuDNN can't find an algorithm for your configuration.

**Common causes and solutions:**

1. **GPU too old**: Check compute capability (need SM70+, SM80+ for SDPA)
2. **Invalid tensor dimensions**: Ensure shapes are valid
3. **Unsupported data type combination**: Use supported types (HALF, BFLOAT16, FLOAT)

```python
# Check GPU capability
import torch
print(torch.cuda.get_device_capability())  # Should be (8, 0) or higher for SDPA
```

### "Tensor dimensions mismatch" error

**Solution:** Ensure tensors match what you used when building the graph:

```python
# Graph was built with shape [8, 64, 56, 56]
# Execution must use the SAME shape
x = torch.randn(8, 64, 56, 56, ...)  # Correct
x = torch.randn(16, 64, 56, 56, ...)  # Wrong - different batch size
```

### Forgot to call `set_output(True)`

```python
# WRONG
with cudnn.Graph() as graph:
    y = graph.conv_fprop(x, w)
    # Missing: y.set_output(True)

# CORRECT
with cudnn.Graph() as graph:
    y = graph.conv_fprop(x, w)
    y.set_output(True)  # Required!
```

## Performance Issues

### Slow first execution

This is normal! cuDNN compiles and optimizes the graph on first run.

**Solution:** Warm up before timing:

```python
# Warmup
for _ in range(3):
    _ = graph(x, w, handle=handle)
torch.cuda.synchronize()

# Now time it
start = time.time()
for _ in range(100):
    _ = graph(x, w, handle=handle)
torch.cuda.synchronize()
print(f"Time: {(time.time()-start)/100*1000:.2f} ms")
```

### Performance worse than expected

**Checklist:**

1. ✅ Using channels-last layout? (`.to(memory_format=torch.channels_last)`)
2. ✅ Using FP16/BF16 for I/O?
3. ✅ Batch size large enough? (>= 32 recommended)
4. ✅ Tensor dimensions aligned? (multiples of 8 for Tensor Cores)
5. ✅ Graph reused, not rebuilt each iteration?

### Memory usage too high

**Solutions:**

1. Use virtual tensors for intermediates (don't mark all tensors as output)
2. Reduce batch size
3. Use lower precision (FP16 instead of FP32)
4. Limit workspace size:
   ```python
   graph.create_execution_plans(
       [cudnn.heur_mode.A],
       max_workspace_size=256 * 1024 * 1024  # 256MB limit
   )
   ```

## SDPA-Specific Issues

### "SDPA requires SM80 or higher"

SDPA (scaled dot-product attention) requires Ampere GPU or newer.

**Check your GPU:**
```python
print(torch.cuda.get_device_capability())
# Needs to be (8, 0) or higher
```

### Wrong output shape from SDPA

**Must set output dimensions:**

```python
output, _ = graph.sdpa(q=Q, k=K, v=V, ...)
output.set_output(True)
output.set_dim(Q.shape)      # Required!
output.set_stride(Q.stride()) # Required!
```

### Numerical differences with PyTorch

Small differences (1e-2 to 1e-3) are expected due to:
- Different algorithm implementations
- Different accumulation order
- Mixed precision effects

```python
# Use appropriate tolerances
torch.testing.assert_close(result, reference, atol=5e-3, rtol=3e-3)
```

## Training Issues

### Need gradients for backward pass

For training, set `is_inference=False` and `generate_stats=True`:

```python
# Forward
output, stats = graph.sdpa(
    q=Q, k=K, v=V,
    is_inference=False,   # Training mode
    generate_stats=True,  # Save for backward
)
stats.set_output(True)  # Save stats!
```

### Backward pass errors

Make sure to:
1. Save stats from forward pass
2. Use same parameters in backward as forward
3. Set output on all gradient tensors

## Debugging Tips

### Enable logging

```bash
export CUDNN_FRONTEND_LOG_INFO=1
export CUDNN_FRONTEND_LOG_FILE=stdout
```

### Print graph info

```python
print(graph)  # Shows graph structure after building
```

### Check workspace size

```python
workspace_size = graph.get_workspace_size()
print(f"Workspace needed: {workspace_size / 1024**2:.1f} MB")
```

### Verify tensor properties

```python
print(f"Shape: {x.shape}")
print(f"Stride: {x.stride()}")
print(f"Dtype: {x.dtype}")
print(f"Device: {x.device}")
print(f"Contiguous: {x.is_contiguous()}")
print(f"Channels-last: {x.is_contiguous(memory_format=torch.channels_last)}")
```

## Getting More Help

1. **GitHub Issues**: [github.com/NVIDIA/cudnn-frontend/issues](https://github.com/NVIDIA/cudnn-frontend/issues)
2. **NVIDIA Forums**: [forums.developer.nvidia.com](https://forums.developer.nvidia.com/)
3. **cuDNN Documentation**: [docs.nvidia.com/deeplearning/cudnn](https://docs.nvidia.com/deeplearning/cudnn)

When reporting issues, include:
- GPU model (`nvidia-smi`)
- cuDNN version (`cudnn.backend_version()`)
- PyTorch version
- Minimal reproducible code
- Full error message

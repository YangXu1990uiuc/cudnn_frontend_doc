# Installation Guide

## Prerequisites

Before installing cuDNN Frontend, make sure you have:

| Requirement | Minimum Version | Recommended |
|-------------|-----------------|-------------|
| NVIDIA GPU | Volta (V100) | Ampere (A100) or newer |
| NVIDIA Driver | 450.0+ | Latest stable |
| CUDA Toolkit | 11.0+ | 12.x |
| Python | 3.8+ | 3.10+ |

!!! tip "Check Your GPU"
    Run `nvidia-smi` to verify your GPU and driver:
    ```bash
    nvidia-smi
    ```
    Look for the GPU name and driver version in the output.

## Installation Methods

### Option 1: pip Install (Recommended for Python)

The easiest way to get started:

```bash
pip install nvidia-cudnn-frontend
```

This single command installs:

- cuDNN Frontend Python bindings
- Required cuDNN backend library
- All Python dependencies

!!! success "That's it!"
    You can now start using cuDNN Frontend in Python.

**Verify the installation:**

```python
import cudnn
print(f"cuDNN Backend Version: {cudnn.backend_version()}")
```

### Option 2: C++ Header-Only Library

cuDNN Frontend for C++ is header-only - no compilation needed!

**Step 1: Clone the repository**

```bash
git clone https://github.com/NVIDIA/cudnn-frontend.git
cd cudnn-frontend
```

**Step 2: Include in your project**

Add the include directory to your compiler flags:

```bash
-I/path/to/cudnn-frontend/include
```

**Step 3: Include the header**

```cpp
#include <cudnn_frontend.h>
```

### Option 3: Build from Source

For development or customization:

**Step 1: Clone and enter the repository**

```bash
git clone https://github.com/NVIDIA/cudnn-frontend.git
cd cudnn-frontend
```

**Step 2: Install Python build dependencies**

```bash
pip install -r requirements.txt
```

**Step 3: Build and install**

```bash
pip install -v .
```

!!! note "Environment Variables"
    You can customize paths with environment variables:

    - `CUDAToolkit_ROOT`: Path to CUDA installation
    - `CUDNN_PATH`: Path to cuDNN installation

## Environment Setup

### Setting Up CUDA Environment

Make sure CUDA is in your path:

=== "Linux/macOS"

    ```bash
    export PATH=/usr/local/cuda/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
    ```

=== "Windows"

    ```powershell
    $env:PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin;$env:PATH"
    ```

### Virtual Environment (Recommended)

We recommend using a virtual environment. **UV is the fastest option:**

=== "uv (Recommended)"

    ```bash
    # Install uv if you haven't
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Create and activate environment
    uv venv cudnn-env
    source cudnn-env/bin/activate  # Linux/macOS

    # Install packages (10-100x faster than pip)
    uv pip install nvidia-cudnn-frontend torch
    ```

=== "venv"

    ```bash
    python -m venv cudnn-env
    source cudnn-env/bin/activate  # Linux/macOS
    cudnn-env\Scripts\activate     # Windows

    pip install nvidia-cudnn-frontend
    ```

=== "conda"

    ```bash
    conda create -n cudnn-env python=3.10
    conda activate cudnn-env

    pip install nvidia-cudnn-frontend
    ```

## Verification

Let's make sure everything is working. Create a file called `test_cudnn.py`:

```python
import cudnn
import torch

# Check versions
print(f"cuDNN Backend Version: {cudnn.backend_version()}")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Create a simple test
    handle = cudnn.create_handle()

    # Test tensor
    x = torch.randn(4, 64, 32, 32, device="cuda", dtype=torch.float16)
    w = torch.randn(128, 64, 3, 3, device="cuda", dtype=torch.float16)

    # Build a simple convolution graph
    with cudnn.Graph(
        io_data_type=cudnn.data_type.HALF,
        compute_data_type=cudnn.data_type.FLOAT,
    ) as graph:
        y = graph.conv_fprop(
            image=x, weight=w,
            padding=[1, 1], stride=[1, 1], dilation=[1, 1]
        )
        y.set_output(True)

    # Execute
    result = graph(x, w, handle=handle)

    print(f"Output shape: {result.shape}")
    print("SUCCESS: cuDNN Frontend is working!")

    cudnn.destroy_handle(handle)
```

Run it:

```bash
python test_cudnn.py
```

Expected output:

```
cuDNN Backend Version: 90300
PyTorch Version: 2.x.x
CUDA Available: True
GPU: NVIDIA GeForce RTX ...
Output shape: torch.Size([4, 128, 32, 32])
SUCCESS: cuDNN Frontend is working!
```

## Troubleshooting

### Common Issues

??? failure "CUDA not found"
    **Error:** `CUDA not available` or `No CUDA GPUs are available`

    **Solution:**

    1. Verify GPU with `nvidia-smi`
    2. Check CUDA installation: `nvcc --version`
    3. Ensure PyTorch CUDA version matches your system:
       ```bash
       pip install torch --index-url https://download.pytorch.org/whl/cu126
       ```

??? failure "cuDNN version mismatch"
    **Error:** `cuDNN version X.Y.Z expected, found A.B.C`

    **Solution:**

    1. Install matching cuDNN version:
       ```bash
       pip install nvidia-cudnn-cu12==9.0.0
       ```
    2. Or update cudnn-frontend:
       ```bash
       pip install --upgrade nvidia-cudnn-frontend
       ```

??? failure "Import errors"
    **Error:** `ModuleNotFoundError: No module named 'cudnn'`

    **Solution:**

    1. Verify installation:
       ```bash
       pip list | grep cudnn
       ```
    2. Reinstall if needed:
       ```bash
       pip uninstall nvidia-cudnn-frontend
       pip install nvidia-cudnn-frontend
       ```

### Getting Help

If you're still having issues:

1. Check the [GitHub Issues](https://github.com/NVIDIA/cudnn-frontend/issues)
2. Search the [NVIDIA Developer Forums](https://forums.developer.nvidia.com/)
3. Open a new issue with:
      - Your GPU model
      - CUDA version
      - Python version
      - Full error message

## Next Steps

Your environment is ready! Let's build your first attention graph.

[Quick Start (SDPA) :material-arrow-right:](llm-quickstart.md){ .md-button .md-button--primary }

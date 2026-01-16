# The CUDA Ecosystem

Before diving deeper into cuDNN Frontend, let's understand where it fits in the larger NVIDIA software stack. This context will help you make better decisions and debug issues more effectively.

## The Big Picture

```mermaid
graph TB
    subgraph "Your Application Layer"
        App[Your AI/ML Application]
    end

    subgraph "Training Frameworks"
        PT[PyTorch]
        TF[TensorFlow]
        JAX[JAX]
        Nemo[Nemo]
    end

    subgraph "Inference Frameworks"
        TRTLLM[TensorRT-LLM]
        vLLM[vLLM]
        SGLang[SGLang]
        TRT[TensorRT]
    end
    
    subgraph "CUDA Libraries"
        FI[FlashInfer]
        cuDNN[cuDNN Frontend/Backend]
        cuBLAS[cuBLAS]
        CUTLASS[CUTLASS]
        NCCL[NCCL]
    end

    subgraph "Runtime"
        CUDA[CUDA Runtime]
    end

    subgraph "Hardware"
        GPU[NVIDIA GPU]
    end

    App --> TRTLLM & vLLM & SGLang
    App --> PT & TF & JAX & Nemo
    App --> TRT

    TRTLLM & vLLM & SGLang --> FI
    FI --> cuDNN
    cuDNN --> cuBLAS
    cuDNN --> CUTLASS

    PT & TF & JAX & Nemo --> cuDNN & cuBLAS & NCCL

    TRT --> CUDA
    cuDNN & CUTLASS --> CUDA
    cuBLAS & NCCL --> CUDA
    CUDA --> GPU

    style cuDNN fill:#c8e6c9,stroke:#2e7d32,stroke-width:3px
    style FI fill:#e1bee7
    style GPU fill:#76ff03
```

## Component Overview

| Component | What It Does |
|-----------|--------------|
| **Training Frameworks** | PyTorch, TensorFlow, JAX, Nemo - high-level APIs for model development |
| **Inference Frameworks** | TensorRT-LLM, vLLM, SGLang - optimized LLM serving; TensorRT - general inference |
| **FlashInfer** | Unified kernel library for LLM inference (MLSys 2025 Best Paper) |
| **cuDNN Frontend** | Modern graph-based API for deep learning ops - **what you're learning!** |
| **cuDNN Backend** | Core C library with optimized GPU kernels |
| **cuBLAS** | Basic linear algebra (GEMM, etc.) |
| **CUTLASS** | Template library for custom CUDA kernels |
| **NCCL** | Multi-GPU communication (AllReduce, etc.) |
| **CUDA Runtime** | Foundation for all GPU computing |

## Next Steps

Now that you understand where cuDNN Frontend fits, learn why it exists and what problems it solves.

[Why cuDNN Frontend? :material-arrow-right:](../getting-started/introduction.md){ .md-button .md-button--primary }

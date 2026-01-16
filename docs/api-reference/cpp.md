# C++ API Reference

This reference covers the **cuDNN Graph API** for C++ - the modern, header-only API for high-performance deep learning. The C++ API is functionally equivalent to Python and is ideal for production inference systems.

!!! info "Graph API Focus"
    This documentation covers the **Graph API** (v1.x). The legacy descriptor-based API is not covered here.

## Setup

cuDNN Frontend for C++ is header-only:

```cpp
#include <cudnn_frontend.h>

namespace fe = cudnn_frontend;
```

## Core Classes

### cudnn_frontend::graph::Graph

The primary class for building computation graphs.

```cpp
namespace cudnn_frontend::graph {

class Graph {
public:
    // Constructor with optional name
    Graph(std::string name = "");

    // Set global data types
    Graph& set_io_data_type(DataType_t dtype);
    Graph& set_intermediate_data_type(DataType_t dtype);
    Graph& set_compute_data_type(DataType_t dtype);

    // Create tensors
    std::shared_ptr<Tensor> tensor(TensorBuilder const&);

    // Operations (see below)

    // Build and execution
    error_t validate();
    error_t build_operation_graph(cudnnHandle_t handle);
    error_t create_execution_plans(std::vector<HeurMode_t> const& modes);
    error_t check_support(cudnnHandle_t handle);
    error_t build_plans(cudnnHandle_t handle);
    error_t execute(cudnnHandle_t handle,
                    VariantPack const& variant_pack,
                    void* workspace);

    // Utilities
    int64_t get_workspace_size() const;
    error_t serialize(std::vector<uint8_t>& data) const;
    static Graph deserialize(std::vector<uint8_t> const& data);
};

}  // namespace cudnn_frontend::graph
```

### Basic Usage

```cpp
#include <cudnn_frontend.h>
#include <cuda_runtime.h>

namespace fe = cudnn_frontend;

int main() {
    // Create cuDNN handle
    cudnnHandle_t handle;
    cudnnCreate(&handle);

    // Create graph
    auto graph = fe::graph::Graph();
    graph.set_io_data_type(fe::DataType_t::HALF)
         .set_compute_data_type(fe::DataType_t::FLOAT);

    // Create input tensors
    auto X = graph.tensor(fe::graph::Tensor_attributes()
        .set_name("X")
        .set_dim({8, 64, 56, 56})
        .set_stride({64*56*56, 1, 64*56, 64})  // NHWC
        .set_data_type(fe::DataType_t::HALF));

    auto W = graph.tensor(fe::graph::Tensor_attributes()
        .set_name("W")
        .set_dim({128, 64, 3, 3})
        .set_stride({64*3*3, 1, 64*3, 64})
        .set_data_type(fe::DataType_t::HALF));

    // Add convolution operation
    auto conv_options = fe::graph::Conv_fprop_attributes()
        .set_padding({1, 1})
        .set_stride({1, 1})
        .set_dilation({1, 1});

    auto Y = graph.conv_fprop(X, W, conv_options);
    Y->set_output(true);

    // Build the graph
    graph.validate();
    graph.build_operation_graph(handle);
    graph.create_execution_plans({fe::HeurMode_t::A, fe::HeurMode_t::FALLBACK});
    graph.check_support(handle);
    graph.build_plans(handle);

    // Allocate memory and execute
    void *x_ptr, *w_ptr, *y_ptr, *workspace;
    // ... allocate CUDA memory ...

    auto workspace_size = graph.get_workspace_size();
    cudaMalloc(&workspace, workspace_size);

    std::unordered_map<std::shared_ptr<fe::graph::Tensor>, void*> variant_pack = {
        {X, x_ptr},
        {W, w_ptr},
        {Y, y_ptr}
    };

    graph.execute(handle, variant_pack, workspace);

    // Cleanup
    cudnnDestroy(handle);
    return 0;
}
```

## Data Types

```cpp
enum class DataType_t {
    FLOAT,       // FP32
    HALF,        // FP16
    BFLOAT16,    // BF16
    FP8_E4M3,    // 8-bit FP (Hopper+)
    FP8_E5M2,    // 8-bit FP for gradients
    INT8,        // Quantized
    INT32,       // Indices
    INT64,       // Large indices
    BOOLEAN      // Masks
};
```

## Tensor Attributes

```cpp
class Tensor_attributes {
public:
    Tensor_attributes& set_name(std::string const& name);
    Tensor_attributes& set_dim(std::vector<int64_t> const& dim);
    Tensor_attributes& set_stride(std::vector<int64_t> const& stride);
    Tensor_attributes& set_data_type(DataType_t dtype);
    Tensor_attributes& set_is_virtual(bool is_virtual);
    Tensor_attributes& set_is_pass_by_value(bool is_pass_by_value);
};
```

## Operations

### Scaled Dot-Product Attention

For LLM/VLM attention layers:

```cpp
class SDPA_attributes {
public:
    SDPA_attributes& set_attn_scale(float scale);
    SDPA_attributes& set_is_inference(bool is_inference);
    SDPA_attributes& set_causal_mask(bool use_causal);
    SDPA_attributes& set_dropout(float probability);
    SDPA_attributes& set_name(std::string const& name);
};

// Returns tuple of (output, stats)
auto [O, Stats] = graph.sdpa(Q, K, V, SDPA_attributes()
    .set_attn_scale(1.0f / sqrtf(head_dim))
    .set_is_inference(true)
    .set_causal_mask(true)
    .set_name("self_attention"));

O->set_output(true);
```

**LLM Inference Example:**

```cpp
// Self-attention for LLM inference
auto create_attention_graph(
    int batch_size, int num_heads, int seq_len, int head_dim,
    cudnnHandle_t handle
) {
    auto graph = fe::graph::Graph();
    graph.set_io_data_type(fe::DataType_t::HALF)
         .set_intermediate_data_type(fe::DataType_t::FLOAT)
         .set_compute_data_type(fe::DataType_t::FLOAT);

    // Q, K, V: [batch, num_heads, seq_len, head_dim]
    auto Q = graph.tensor(fe::graph::Tensor_attributes()
        .set_name("Q")
        .set_dim({batch_size, num_heads, seq_len, head_dim})
        .set_stride({num_heads*seq_len*head_dim, seq_len*head_dim, head_dim, 1})
        .set_data_type(fe::DataType_t::HALF));

    auto K = graph.tensor(fe::graph::Tensor_attributes()
        .set_name("K")
        .set_dim({batch_size, num_heads, seq_len, head_dim})
        .set_stride({num_heads*seq_len*head_dim, seq_len*head_dim, head_dim, 1})
        .set_data_type(fe::DataType_t::HALF));

    auto V = graph.tensor(fe::graph::Tensor_attributes()
        .set_name("V")
        .set_dim({batch_size, num_heads, seq_len, head_dim})
        .set_stride({num_heads*seq_len*head_dim, seq_len*head_dim, head_dim, 1})
        .set_data_type(fe::DataType_t::HALF));

    // SDPA with causal mask
    auto [O, Stats] = graph.sdpa(Q, K, V,
        fe::graph::SDPA_attributes()
            .set_attn_scale(1.0f / sqrtf(static_cast<float>(head_dim)))
            .set_is_inference(true)
            .set_causal_mask(true));

    O->set_output(true);
    O->set_dim({batch_size, num_heads, seq_len, head_dim});
    O->set_stride({num_heads*seq_len*head_dim, seq_len*head_dim, head_dim, 1});

    // Build
    graph.validate();
    graph.build_operation_graph(handle);
    graph.create_execution_plans({fe::HeurMode_t::A});
    graph.check_support(handle);
    graph.build_plans(handle);

    return std::make_tuple(graph, Q, K, V, O);
}
```

### Matrix Multiplication

```cpp
class Matmul_attributes {
public:
    Matmul_attributes& set_compute_data_type(DataType_t dtype);
    Matmul_attributes& set_name(std::string const& name);
};

auto C = graph.matmul(A, B, Matmul_attributes()
    .set_compute_data_type(fe::DataType_t::FLOAT)
    .set_name("projection"));
```

### Normalization

#### Layer Normalization

```cpp
class Layernorm_attributes {
public:
    Layernorm_attributes& set_epsilon(float eps);
    Layernorm_attributes& set_name(std::string const& name);
};

auto [Y, Mean, InvVar] = graph.layernorm(X, Scale, Bias,
    Layernorm_attributes()
        .set_epsilon(1e-5f)
        .set_name("pre_attn_norm"));

Y->set_output(true);
```

#### RMS Normalization

```cpp
class Rmsnorm_attributes {
public:
    Rmsnorm_attributes& set_epsilon(float eps);
    Rmsnorm_attributes& set_name(std::string const& name);
};

auto [Y, InvRms] = graph.rmsnorm(X, Scale,
    Rmsnorm_attributes()
        .set_epsilon(1e-5f)
        .set_name("rms_norm"));
```

### Convolution (for VLM vision encoders)

```cpp
class Conv_fprop_attributes {
public:
    Conv_fprop_attributes& set_padding(std::vector<int64_t> const& padding);
    Conv_fprop_attributes& set_stride(std::vector<int64_t> const& stride);
    Conv_fprop_attributes& set_dilation(std::vector<int64_t> const& dilation);
    Conv_fprop_attributes& set_name(std::string const& name);
};

auto Y = graph.conv_fprop(X, W, Conv_fprop_attributes()
    .set_padding({1, 1})
    .set_stride({1, 1})
    .set_dilation({1, 1})
    .set_name("vision_conv"));
```

### Pointwise Operations

```cpp
// ReLU
auto Y = graph.relu(X);

// GELU
auto Y = graph.gelu(X);

// SiLU/Swish
auto Y = graph.silu(X);

// Addition
auto Y = graph.add(A, B);

// Multiplication
auto Y = graph.mul(A, B);
```

## Build Process

```cpp
// 1. Validate graph structure
auto status = graph.validate();
if (status.is_bad()) {
    std::cerr << "Validation failed: " << status.get_message() << std::endl;
    return -1;
}

// 2. Build operation graph
graph.build_operation_graph(handle);

// 3. Create execution plans with heuristics
graph.create_execution_plans({
    fe::HeurMode_t::A,
    fe::HeurMode_t::FALLBACK
});

// 4. Check hardware support
status = graph.check_support(handle);
if (status.is_bad()) {
    std::cerr << "Not supported: " << status.get_message() << std::endl;
    return -1;
}

// 5. Build selected plans
graph.build_plans(handle);
```

## Execution

```cpp
// Get workspace size
int64_t workspace_size = graph.get_workspace_size();

// Allocate workspace
void* workspace;
cudaMalloc(&workspace, workspace_size);

// Create variant pack (tensor -> device pointer mapping)
std::unordered_map<std::shared_ptr<fe::graph::Tensor>, void*> variant_pack = {
    {Q, q_device_ptr},
    {K, k_device_ptr},
    {V, v_device_ptr},
    {O, o_device_ptr}
};

// Execute
auto status = graph.execute(handle, variant_pack, workspace);
cudaDeviceSynchronize();
```

## Serialization

```cpp
// Serialize graph to bytes
std::vector<uint8_t> serialized_data;
graph.serialize(serialized_data);

// Save to file
std::ofstream file("attention_graph.bin", std::ios::binary);
file.write(reinterpret_cast<char*>(serialized_data.data()),
           serialized_data.size());
file.close();

// Later: deserialize
std::ifstream file("attention_graph.bin", std::ios::binary);
std::vector<uint8_t> loaded_data(
    std::istreambuf_iterator<char>(file),
    std::istreambuf_iterator<char>());

auto loaded_graph = fe::graph::Graph::deserialize(loaded_data);
// Ready to execute immediately!
```

## Complete LLM Attention Example

```cpp
#include <cudnn_frontend.h>
#include <cuda_runtime.h>
#include <cmath>

namespace fe = cudnn_frontend;

class LLMAttention {
public:
    LLMAttention(int batch, int heads, int seq_len, int head_dim)
        : batch_(batch), heads_(heads), seq_len_(seq_len), head_dim_(head_dim)
    {
        cudnnCreate(&handle_);
        build_graph();
    }

    ~LLMAttention() {
        cudaFree(workspace_);
        cudnnDestroy(handle_);
    }

    void forward(void* q_ptr, void* k_ptr, void* v_ptr, void* o_ptr) {
        std::unordered_map<std::shared_ptr<fe::graph::Tensor>, void*> variant_pack = {
            {Q_, q_ptr}, {K_, k_ptr}, {V_, v_ptr}, {O_, o_ptr}
        };
        graph_.execute(handle_, variant_pack, workspace_);
    }

private:
    void build_graph() {
        graph_.set_io_data_type(fe::DataType_t::HALF)
              .set_intermediate_data_type(fe::DataType_t::FLOAT)
              .set_compute_data_type(fe::DataType_t::FLOAT);

        std::vector<int64_t> dims = {batch_, heads_, seq_len_, head_dim_};
        std::vector<int64_t> strides = {
            heads_ * seq_len_ * head_dim_,
            seq_len_ * head_dim_,
            head_dim_,
            1
        };

        Q_ = graph_.tensor(fe::graph::Tensor_attributes()
            .set_name("Q").set_dim(dims).set_stride(strides)
            .set_data_type(fe::DataType_t::HALF));

        K_ = graph_.tensor(fe::graph::Tensor_attributes()
            .set_name("K").set_dim(dims).set_stride(strides)
            .set_data_type(fe::DataType_t::HALF));

        V_ = graph_.tensor(fe::graph::Tensor_attributes()
            .set_name("V").set_dim(dims).set_stride(strides)
            .set_data_type(fe::DataType_t::HALF));

        float scale = 1.0f / sqrtf(static_cast<float>(head_dim_));
        auto [O, Stats] = graph_.sdpa(Q_, K_, V_,
            fe::graph::SDPA_attributes()
                .set_attn_scale(scale)
                .set_is_inference(true)
                .set_causal_mask(true));

        O_ = O;
        O_->set_output(true);
        O_->set_dim(dims);
        O_->set_stride(strides);

        graph_.validate();
        graph_.build_operation_graph(handle_);
        graph_.create_execution_plans({fe::HeurMode_t::A});
        graph_.check_support(handle_);
        graph_.build_plans(handle_);

        cudaMalloc(&workspace_, graph_.get_workspace_size());
    }

    int batch_, heads_, seq_len_, head_dim_;
    cudnnHandle_t handle_;
    fe::graph::Graph graph_;
    std::shared_ptr<fe::graph::Tensor> Q_, K_, V_, O_;
    void* workspace_ = nullptr;
};
```

## CMake Integration

```cmake
cmake_minimum_required(VERSION 3.18)
project(cudnn_frontend_example CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Find cuDNN
find_path(CUDNN_INCLUDE_DIR cudnn.h
    HINTS ${CUDNN_PATH}/include)
find_library(CUDNN_LIBRARY cudnn
    HINTS ${CUDNN_PATH}/lib64)

# Add executable
add_executable(attention_example main.cpp)

target_include_directories(attention_example PRIVATE
    ${CUDNN_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/cudnn-frontend/include)

target_link_libraries(attention_example
    CUDA::cudart
    ${CUDNN_LIBRARY})
```

## Next Steps

See common patterns and best practices.

[Common Patterns :material-arrow-right:](patterns.md){ .md-button .md-button--primary }

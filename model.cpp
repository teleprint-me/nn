/*
 * model.cpp
 */

#include "ggml.h"

#include <cstdio>

void verify_tensor_creation(
    struct ggml_context* ctx, struct ggml_tensor* tensor
) {
    if (!tensor) {
        fprintf(stderr, "Failed to create ggml tensor\n");
        return;
    }
}

int main() {
    // Initialize the ggml context with a predefined memory size
    ggml_init_params params = {
        .mem_size   = 16 * 1024 * 1024, // 16 MB
        .mem_buffer = NULL,             // Let ggml manage the memory allocation
        .no_alloc   = false             // Allocate memory for the tensor data
    };

    ggml_context* ctx = ggml_init(params);
    if (!ctx) {
        fprintf(stderr, "Failed to initialize ggml context\n");
        return 1;
    }

    // Define tensor dimensions
    int64_t rows = 2;
    int64_t cols = 3;

    // Create a 2D tensor with half precision (F16)
    struct ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, rows, cols);
    verify_tensor_creation(ctx, a);
    ggml_set_f32(a, 0); // uniformly zero-initialize the tensor

    struct ggml_tensor* b = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, rows, cols);
    verify_tensor_creation(ctx, b);
    ggml_set_f32(b, 0); // uniformly zero-initialize the tensor

    // todo: set the computation graph, otherwise nothing happens

    // Perform matrix multiplication using ggml_mul()
    struct ggml_tensor* x = ggml_mul(ctx, a, b);
    verify_tensor_creation(ctx, x);

    // Tensor addition operation
    struct ggml_tensor* f = ggml_add(ctx, ggml_mul(ctx, a, x), b);

    ggml_free(ctx);

    return 0;
}

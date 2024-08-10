/*
    model.cpp
*/

#include "ggml.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>

void verify_tensor_creation(
    struct ggml_context* ctx, struct ggml_tensor* tensor
) {
    if (!tensor) {
        // Handle tensor creation failure
        std::cerr << "Failed to create ggml tensor" << std::endl;
        ggml_free(ctx);
        throw std::runtime_error("Tensor creation failed");
    }
}

int main() {
    // Initialize the ggml context with a predefined memory size
    struct ggml_init_params params = {
        .mem_size   = 16 * 1024 * 1024, // 16 MB
        .mem_buffer = NULL,             // Let ggml manage the memory allocation
    };

    struct ggml_context* ctx = ggml_init(params);
    if (!ctx) {
        // Handle potential initialization failure
        fprintf(stderr, "Failed to initialize ggml context\n");
        return 1;
    }

    // Define tensor dimensions
    int64_t rows = 4; // number of elements
    int64_t cols = 4;

    // Create a 2D tensor with half precision (F16)
    struct ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, rows, cols);
    verify_tensor_creation(ctx, a);

    struct ggml_tensor* b = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, rows, cols);
    verify_tensor_creation(ctx, b);

    // originally labeled as x2? why? seems to have no significant meaning.
    // x2 may have been a randomly chosen variable name. x may be sufficient to
    // avoid confusion.
    // TODO: Backward pass is currently unsupported
    struct ggml_tensor* x = ggml_mul(ctx, a, b);

    struct ggml_tensor* f = ggml_add(ctx, ggml_mul(ctx, a, x), b);

    // Clean up and free resources
    ggml_free(ctx);

    return 0;
}

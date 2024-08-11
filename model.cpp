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

void print_tensor_info(struct ggml_tensor* tensor, enum ggml_type type) {
    if (!tensor) {
        fprintf(stderr, "Tensor is NULL\n");
        return;
    }

    printf("Tensor Info:\n");
    printf("Name: %s\n", tensor->name);
    printf("Type: %d\n", tensor->type);
    printf("Dimensions: ");
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        printf("%lld ", tensor->ne[i]);
    }
    printf("\n");

    printf("Strides: ");
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        printf("%zu ", tensor->nb[i]);
    }
    printf("\n");

    printf("Data Pointer: %p\n", tensor->data);
    printf("View Source: %p\n", tensor->view_src);
    printf("View Offset: %zu\n", tensor->view_offs);

    // If the tensor has data, print the first few elements (assuming it's a
    // float tensor)
    if (tensor->data && tensor->type == type) {
        printf("First few elements:\n");
        float* data_f32 = (float*) tensor->data;
        for (int i = 0; i < 10 && i < tensor->ne[0]; ++i) {
            printf("%f ", data_f32[i]);
        }
        printf("\n");
    }

    printf("----\n");
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

    // Create a 2D tensor and print its information
    struct ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 4);
    ggml_set_f32(a, 42.0f); // Set all elements to 42.0
    print_tensor_info(a, GGML_TYPE_F32);

    // Create a 2D tensor and print its information
    struct ggml_tensor* b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 4);
    ggml_set_f32(b, 0.0f); // Set all elements to 42.0
    print_tensor_info(b, GGML_TYPE_F32);

    // // todo: set the computation graph, otherwise nothing happens

    // // Perform matrix multiplication using ggml_mul()
    // struct ggml_tensor* x = ggml_mul(ctx, a, b);
    // verify_tensor_creation(ctx, x);

    // // Tensor addition operation
    // struct ggml_tensor* f = ggml_add(ctx, ggml_mul(ctx, a, x), b);

    ggml_free(ctx);

    return 0;
}

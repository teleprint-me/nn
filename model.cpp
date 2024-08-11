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
        for (int i = 0; i < GGML_MAX_SRC && i < tensor->ne[0]; ++i) {
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
    int64_t ne0 = 3; // number elements with in a row
    int64_t ne1 = 4; // number of elements iwthin a column

    // Define the tensors to be used within the computation graph
    struct ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, ne1);
    ggml_set_name(a, "a"); // label the tensor for identifcation

    struct ggml_tensor* b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, ne1);
    ggml_set_name(b, "b");

    // Define operations for each node within the computation graph
    struct ggml_tensor* x = ggml_mul(ctx, a, b);
    ggml_set_name(x, "x");

    struct ggml_tensor* f = ggml_add(ctx, ggml_mul(ctx, a, x), b);
    ggml_set_name(f, "f");

    // Build the computation graph
    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, f);

    // Initialize the tensors and computation graph
    ggml_set_f32(a, 2.0f); // Initialize elements in a to 2.0f
    print_tensor_info(a, GGML_TYPE_F32);

    ggml_set_f32(b, 1.0f); // Initialize elements in b to 1.0f
    print_tensor_info(b, GGML_TYPE_F32);

    ggml_set_f32(x, 3.0f); // Initialize elements in x to 3.0f
    print_tensor_info(x, GGML_TYPE_F32);
    // same as ggml_graph_compute() but the work data is allocated as a part of
    // the context note: the drawback of this API is that you must have ensured
    // that the context has enough memory for the work data
    // ggml_status ggml_graph_compute_with_ctx(
    //     ggml_context *ctx, ggml_cgraph *cgraph, int n_threads
    // )
    ggml_graph_compute_with_ctx(ctx, gf, 8);
    print_tensor_info(f, GGML_TYPE_F32);

    ggml_free(ctx);

    return 0;
}

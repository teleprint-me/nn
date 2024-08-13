/*
 * model.cpp
 *
 * Reference docs/model.md for detailed information
 */

#include "ggml.h"
#include "nn.h"

#include <cstdio>

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
    int64_t ne0 = 3; // number of elements within a row
    int64_t ne1 = 4; // number of elements within a column

    // Define the tensors to be used within the computation graph
    struct ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, ne1);
    ggml_set_name(a, "a"); // label the tensor for identification

    struct ggml_tensor* b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, ne1);
    ggml_set_name(b, "b");

    // Initialize the input tensors
    ggml_set_f32(a, 2.0f); // Initialize elements in a to 2.0f
    ggml_set_f32(b, 1.0f); // Initialize elements in b to 1.0f

    // Print initialized tensors
    print_tensor_info(a, GGML_TYPE_F32);
    print_tensor_info(b, GGML_TYPE_F32);

    // Define operations for each node within the computation graph
    // x = a * b
    struct ggml_tensor* x = ggml_mul(ctx, a, b);
    ggml_set_name(x, "x");

    // f = a * x + b
    struct ggml_tensor* f = ggml_add(ctx, ggml_mul(ctx, a, x), b);
    ggml_set_name(f, "f");

    // Build the computation graph
    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, f);

    // Compute the graph
    ggml_graph_compute_with_ctx(ctx, gf, 8); // Using 8 threads

    // Print the output tensor after computation
    print_tensor_info(x, GGML_TYPE_F32); // --> 2.0f
    print_tensor_info(f, GGML_TYPE_F32); // --> 5.0f

    // Clean up
    ggml_free(ctx);

    return 0;
}

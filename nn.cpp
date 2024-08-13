#include "nn.h"

#include <random>

// initialization related tools
void he_initialization(ggml_tensor* tensor, uint32_t input_dim) {
    std::random_device              rd;
    std::mt19937                    gen(rd());
    std::normal_distribution<float> dist(0.0f, std::sqrt(2.0f / input_dim));

    float* data = (float*) tensor->data;
    for (int i = 0; i < ggml_nelements(tensor); ++i) {
        data[i] = dist(gen);
    }
}

// Define a helper function to set tensor values manually
void set_tensor_data_f32(
    ggml_tensor* tensor, float* data, int64_t rows, int64_t cols
) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            float* dst = (float*) ((char*) tensor->data + i * tensor->nb[1]
                                   + j * tensor->nb[0]);
            *dst       = data[i * cols + j];
        }
    }
}

// verification related tools
void verify_tensor_creation(
    struct ggml_context* ctx, struct ggml_tensor* tensor
) {
    if (!tensor) {
        fprintf(stderr, "Failed to create ggml tensor\n");
        return;
    }
}

// print related tools
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

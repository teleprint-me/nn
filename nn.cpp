/*
 * nn.cpp
 */

#include "nn.h"

#include <random>

// initialization related tools
void he_initialization(ggml_tensor* tensor, uint32_t input_dim) {
    std::random_device              rd;
    std::mt19937                    gen(rd());
    std::normal_distribution<float> dist(0.0f, std::sqrt(2.0f / input_dim));

    float* data = (float*) tensor->data;
    for (int64_t i = 0; i < ggml_nelements(tensor); ++i) {
        data[i] = dist(gen);
    }
}

// Define a helper function to set tensor values manually
void set_tensor_data_f32(ggml_tensor* tensor, float* data, int64_t rows, int64_t cols) {
    for (int64_t i = 0; i < rows; ++i) {
        for (int64_t j = 0; j < cols; ++j) {
            float* dst = (float*) ((char*) tensor->data + i * tensor->nb[1] + j * tensor->nb[0]);
            *dst       = data[i * cols + j];
        }
    }
}

// verification related tools
void verify_tensor_creation(struct ggml_context* ctx, struct ggml_tensor* tensor) {
    if (!tensor) {
        fprintf(stderr, "Failed to create ggml tensor\n");
        return;
    }
}

// print related tools
void print_tensor_info(struct ggml_tensor* tensor, int64_t max_elements) {
    if (!tensor) {
        fprintf(stderr, "Tensor is NULL\n");
        return;
    }

    printf("Tensor Info:\n");
    printf("Name: %s\n", tensor->name);
    printf("Type: %zu -> %s\n", tensor->type, ggml_type_name(tensor->type));

    printf("Dimensions: ");
    for (int64_t i = 0; i < GGML_MAX_DIMS && tensor->ne[i] > 0; ++i) {
        printf("%lld ", tensor->ne[i]);
    }
    printf("\n");

    printf("Strides: ");
    for (int64_t i = 0; i < GGML_MAX_DIMS && tensor->nb[i] > 0; ++i) {
        printf("%zu ", tensor->nb[i]);
    }
    printf("\n");

    printf("Data Pointer: %p\n", tensor->data);
    printf("View Source: %p\n", tensor->view_src);
    printf("View Offset: %zu\n", tensor->view_offs);

    if (tensor->data) {
        int64_t num_elements = ggml_nelements(tensor);
        if (1 > max_elements) {
            max_elements = num_elements;
        }
        printf("Num elements: %i\n", num_elements);
        printf("Max elements: %i\n", max_elements);

        switch (tensor->type) {
            case GGML_TYPE_F32:
                {
                    float* data_f32 = (float*) tensor->data;
                    for (int64_t i = 0; i < num_elements && i < max_elements; ++i) {
                        printf("%f ", data_f32[i]);
                    }
                    printf("\n");
                    break;
                }
            case GGML_TYPE_F16:
                {
                    uint16_t* data_f16 = (uint16_t*) tensor->data;
                    for (int64_t i = 0; i < num_elements && i < max_elements; ++i) {
                        float data_f32 = ggml_fp16_to_fp32(data_f16[i]);
                        printf("%f ", data_f32);
                    }
                    printf("\n");
                    break;
                }
            // Handle other types similarly...
            default:
                printf("Unsupported tensor data type\n");
                break;
        }
    }

    printf("----\n");
}

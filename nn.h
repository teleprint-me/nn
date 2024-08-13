#ifndef NN_H
#define NN_H

#include "ggml.h"

// initialization related tools
void he_initialization(ggml_tensor* tensor, uint32_t input_dim);
void set_tensor_data_f32(ggml_tensor* tensor, float* data, int64_t rows, int64_t cols);

// verification related tools
void verify_tensor_creation(struct ggml_context* ctx, struct ggml_tensor* tensor);

// print related tools
void print_tensor_info(struct ggml_tensor* tensor, int64_t max_elements);

#endif // NN_H

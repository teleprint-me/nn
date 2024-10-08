/*
 * xor.cpp
 *
 * This file is intended as a test bed for implementing a simple XOR model
 * using the GGML library. The XOR model will demonstrate basic forward and
 * backward computation using GGML's tensor operations and computation graph.
 *
 * Key components to be implemented:
 * - Initialization of tensors for the XOR problem
 * - Definition of the forward pass computation
 * - Implementation of the backward pass for training
 * - Basic error handling and validation
 *
 * Future enhancements:
 * - Integrate advanced features such as custom activation functions
 * - Optimize memory usage and computation efficiency
 * - Explore additional machine learning tasks and models
 *
 * Note: This file is currently a stub and will be developed further as
 * part of the ongoing exploration and application of the GGML library.
 */

#include "ggml.h"
#include "nn.h"

#include <cstring>
#include <random>

struct xor_hparams {
    uint32_t n_input  = 2; // XOR has 2 input features
    uint32_t n_hidden = 4; // Number of hidden units (adjustable)
    uint32_t n_output = 1; // XOR has 1 output
};

struct xor_layer {
    struct ggml_tensor* weights;
    struct ggml_tensor* biases;
};

struct xor_model {
    struct ggml_context* ctx = NULL;
    struct xor_hparams   hparams;

    struct xor_layer input_layer;  // Weights and biases for input to hidden
    struct xor_layer hidden_layer; // Weights and biases for hidden to output

    // std::vector<struct xor_layer> layers;
};

void init_xor_input_layers(xor_model* model) {
    // the models neurons will have 2 inputs and 1 ouput
    model->input_layer.weights
        = ggml_new_tensor_2d(model->ctx, GGML_TYPE_F32, model->hparams.n_hidden, model->hparams.n_input);
    ggml_set_name(model->input_layer.weights, "xor.input_layer.weights");

    model->input_layer.biases = ggml_new_tensor_2d(model->ctx, GGML_TYPE_F32, model->hparams.n_hidden, 1);
    ggml_set_name(model->input_layer.biases, "xor.input_layer.biases");
}

void init_xor_hidden_layers(xor_model* model) {
    // route the outputs to the hidden neurons
    model->hidden_layer.weights
        = ggml_new_tensor_2d(model->ctx, GGML_TYPE_F32, model->hparams.n_output, model->hparams.n_hidden);
    ggml_set_name(model->hidden_layer.weights, "xor.hidden_layer.weights");

    model->hidden_layer.biases = ggml_new_tensor_2d(model->ctx, GGML_TYPE_F32, model->hparams.n_output, 1);
    ggml_set_name(model->hidden_layer.biases, "xor.hidden_layer.biases");
}

struct xor_model init_xor_model(void) {
    // we allocate to the stack for simplicity
    struct xor_model model;

    ggml_init_params params = {.mem_size = 16 * 1024 * 1024, .mem_buffer = NULL, .no_alloc = false};

    model.ctx = ggml_init(params);
    init_xor_input_layers(&model);
    init_xor_hidden_layers(&model);

    return model;
}

ggml_tensor* init_xor_input_tensor(struct xor_model model) {
    // define xor input data
    float x[4][2] = {{0.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 0.0f}, {1.0f, 1.0f}};

    // Create an input tensor and copy the data
    ggml_tensor* input = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, /* rows */ 4, /* cols */ 2);
    // name the tensor for identification
    ggml_set_name(input, "xor.input_tensor.data");
    // add input data to the tensor
    memcpy(input->data, x, sizeof(x));
    // output initialized tensor data
    print_tensor_info(input, /* max_elements */ -1);

    return input;
}

int main(void) {
    // Initialization
    xor_model model = init_xor_model();
    he_initialization(model.input_layer.weights, model.hparams.n_input);
    he_initialization(model.hidden_layer.weights, model.hparams.n_hidden);

    // float output_label[4][1] = {{0.0f}, {1.0f}, {1.0f}, {0.0f}}; // y

    // Create an input tensor and copy the data
    ggml_tensor* input = init_xor_input_tensor(model);

    // Forward pass
    ggml_tensor* input_mul_weights = ggml_mul_mat(model.ctx, model.input_layer.weights, input);
    ggml_tensor* hidden            = ggml_add(model.ctx, input_mul_weights, model.input_layer.biases);

    // Apply ReLU activation
    hidden = ggml_relu(model.ctx, hidden);

    // Hidden layer
    ggml_tensor* hidden_mul_weights = ggml_mul_mat(model.ctx, model.hidden_layer.weights, hidden);
    // output should be 4x1
    ggml_tensor* output             = ggml_add(model.ctx, hidden_mul_weights, model.hidden_layer.biases);

    // Build and compute the forward graph
    struct ggml_cgraph* gf = ggml_new_graph(model.ctx);
    ggml_build_forward_expand(gf, output);
    ggml_graph_compute_with_ctx(model.ctx, gf, 8);

    // Print output
    float* output_data = (float*) output->data;
    for (int i = 0; i < 4; ++i) {
        printf("Output %d: %f\n", i, output_data[i]);
    }

    // Cleanup
    ggml_free(model.ctx);

    return 0;
}

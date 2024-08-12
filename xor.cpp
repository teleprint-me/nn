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
    struct xor_layer     input_layer; // Weights and biases for input to hidden
    struct xor_layer hidden_layer;    // Weights and biases for hidden to output
};

xor_model init_xor_model(void) {
    // we allocate to the stack for simplicity, though we could manually
    // allocate memory if we chose to. GGML does not have anything builtin to
    // handle this allocation, so we would be responsible for its management,
    // hence why we opt for the stack instead
    xor_model model;

    ggml_init_params params
        = {.mem_size = 16 * 1024 * 1024, .mem_buffer = NULL, .no_alloc = false};

    model.ctx = ggml_init(params);

    // initialize the input layers
    // the models neurons will have 2 inputs and 1 ouput
    model.input_layer.weights = ggml_new_tensor_2d(
        model.ctx, GGML_TYPE_F32, model.hparams.n_hidden, model.hparams.n_input
    );
    model.input_layer.biases = ggml_new_tensor_2d(
        model.ctx, GGML_TYPE_F32, model.hparams.n_hidden, 1
    );

    // initialize the hidden layers
    // route the outputs to the hidden neurons
    model.hidden_layer.weights = ggml_new_tensor_2d(
        model.ctx, GGML_TYPE_F32, model.hparams.n_output, model.hparams.n_hidden
    );
    model.hidden_layer.biases = ggml_new_tensor_2d(
        model.ctx, GGML_TYPE_F32, model.hparams.n_output, 1
    );

    return model;
}

int main(void) {
    // implementation steps:
    //   1. initialization
    xor_model model = init_xor_model();

    //   2. forward pass
    //   3. backward pass
    //   4. error handling

    return 0;
}

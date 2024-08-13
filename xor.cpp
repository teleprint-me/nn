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
    struct xor_layer     input_layer; // Weights and biases for input to hidden
    struct xor_layer hidden_layer;    // Weights and biases for hidden to output
};

void init_xor_input_layers(xor_model* model) {
    // the models neurons will have 2 inputs and 1 ouput
    model->input_layer.weights = ggml_new_tensor_2d(
        model->ctx,
        GGML_TYPE_F32,
        model->hparams.n_hidden,
        model->hparams.n_input
    );
    ggml_set_name(model->input_layer.weights, "xor.input_layer.weights");

    model->input_layer.biases = ggml_new_tensor_2d(
        model->ctx, GGML_TYPE_F32, model->hparams.n_hidden, 1
    );
    ggml_set_name(model->input_layer.biases, "xor.input_layer.biases");
}

void init_xor_hidden_layers(xor_model* model) {
    // route the outputs to the hidden neurons
    model->hidden_layer.weights = ggml_new_tensor_2d(
        model->ctx,
        GGML_TYPE_F32,
        model->hparams.n_output,
        model->hparams.n_hidden
    );
    ggml_set_name(model->hidden_layer.weights, "xor.hidden_layer.weights");

    model->hidden_layer.biases = ggml_new_tensor_2d(
        model->ctx, GGML_TYPE_F32, model->hparams.n_output, 1
    );
    ggml_set_name(model->hidden_layer.biases, "xor.hidden_layer.biases");
}

xor_model init_xor_model(void) {
    // we allocate to the stack for simplicity, though we could manually
    // allocate memory if we chose to. GGML does not have anything builtin to
    // handle this allocation, so we would be responsible for its management,
    // hence why we opt for the stack instead
    xor_model model;

    ggml_init_params params
        = {.mem_size = 16 * 1024 * 1024, .mem_buffer = NULL, .no_alloc = false};

    model.ctx = ggml_init(params);
    init_xor_input_layers(&model);
    init_xor_hidden_layers(&model);

    return model;
}

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

int main(void) {
    // 1. Initialization
    xor_model model = init_xor_model();
    he_initialization(model.input_layer.weights, model.hparams.n_input);
    he_initialization(model.hidden_layer.weights, model.hparams.n_hidden);

    // 2. Define the XOR inputs
    int64_t rows             = 4;
    int64_t cols             = 2;
    // i forgot to update this variable name like a dimwit
    float   input_data[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};

    // 3. Create an input tensor and copy the data
    ggml_tensor* input
        = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, cols, rows);
    ggml_set_name(input, "xor.input");
    // using memcpy is fine here, though we can use our custom
    // set_tensor_data_f32 function as well
    memcpy(input->data, input_data, sizeof(input_data));

    // 4. Forward pass - Input layer: (input * weights) + biases
    ggml_tensor* input_mul_weights
        = ggml_mul_mat(model.ctx, model.input_layer.weights, input);
    ggml_tensor* hidden
        = ggml_add(model.ctx, input_mul_weights, model.input_layer.biases);

    // 5. Apply ReLU activation
    hidden = ggml_relu(model.ctx, hidden);

    // 6. Hidden layer: (hidden * weights) + biases
    ggml_tensor* hidden_mul_weights
        = ggml_mul_mat(model.ctx, model.hidden_layer.weights, hidden);
    ggml_tensor* output
        = ggml_add(model.ctx, hidden_mul_weights, model.hidden_layer.biases);

    // At this point, `output` contains the results of the forward pass.
    // Output the final results to verify correctness.

    // we have to create a computation graph
    struct ggml_cgraph* gf = ggml_new_graph(model.ctx);
    // Build the forward computation graph
    // note that there is no ggml_build_forward function defined
    // ggml_build_forward_expand is a wrapper around ggml_build_forward_impl
    // ggml_build_forward_impl is pretty straightforward
    ggml_build_forward_expand(gf, output);

    // For now, let's just print the output tensor data
    // there is a ggml_graph_compute, but we use ggml_graph_compute_with_ctx
    // because it implements a context plan (struct ggml_cplan * cplan) for us
    // which is required by the backend. This is quite involved and we can
    // revisit this in another example in the future. We will use
    // ggml_graph_compute_with_ctx for the sake of simplicity
    ggml_graph_compute_with_ctx(model.ctx, gf, 8); // Compute the graph
    // we already created a helper function for this, but this is fine for now
    float* output_data = (float*) output->data;
    for (int i = 0; i < 4; ++i) {
        printf("Output %d: %f\n", i, output_data[i]);
    }

    // Cleanup GGML context
    ggml_free(model.ctx);

    return 0;
}

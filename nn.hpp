#pragma once

#ifndef NN_HPP_INCLUDED
#define NN_HPP_INCLUDED

#define assert(cond)             \
  do {                           \
    if (!(cond)) __debugbreak(); \
  } while (false)

#include <vector>
#include <fstream>

#include "matrix.hpp"
#include "layer.hpp"

struct NN {
    matrix_t learn_rate = 0.01;
    std::vector<Layer> layers;
    std::vector<std::string> output_labels;

    int trained = 0;
    int data_index = 0;

    NN();
    NN(const std::vector<int>& config, const std::vector<std::string>& output_labels);

    NN_Matrix& get_outputs();

    void forward(const NN_Matrix& input);
    void backprop(const NN_Matrix& expected);
};

NN::NN() {};

NN::NN(const std::vector<int>& config, const std::vector<std::string>& output_labels)
    : output_labels(output_labels) {
        assert(config.size() >= 1);
        assert(output_labels.size() == config.at(config.size() - 1));

        for (size_t i = 0; i < config.size(); i++) {
            int neurons_count = config[i];
            if (i == 0) {
              layers.push_back(Layer(neurons_count));
            } else {
              Layer& prev = layers.at(layers.size() - 1);
              layers.push_back(prev.next_layer(neurons_count));
            }
        }

        for (Layer& layer : layers) {
            layer.weights.randomize(-.5, .5);
        }
    }

NN_Matrix& NN::get_outputs() {
    return layers[layers.size() - 1].outputs;
}


void NN::forward(const NN_Matrix& input) {
    layers[0].outputs = input;
    for (size_t i = 1; i < layers.size(); i++) {
        Layer& curr = layers[i];
        Layer& prev = layers[i - 1];
        Layer::forward(curr, prev);
    }
}

void NN::backprop(const NN_Matrix& expected) {
    NN_Matrix& output = layers[layers.size() - 1].outputs;
    assert(expected.rows() == output.rows() && expected.cols() == output.cols());

  // delta_out = out - exp
  // delta_hidden = w.trans() * next_delta x (a * (1-a))
  //
  // curr_b += -learn_rate * curr_delta
  // prev_w += -learn_rate * (curr_delta.trans() * prev_active)

    NN_Matrix delta = output - expected;
    for (size_t i = layers.size() - 1; i > 0; i--) {
        Layer& curr = layers[i];
        Layer& prev = layers[i - 1];

        curr.biased += (delta * (-learn_rate));
        prev.weights += (prev.outputs.transpose() * delta) * (-learn_rate);

        // sigmoid_derivative = (a * (1 - a));
        NN_Matrix one = NN_Matrix(prev.outputs.rows(), prev.outputs.cols(), 1);
        NN_Matrix sigmoid_derivative = prev.outputs.multiply(one - prev.outputs);

        // delta_next = (delta * prev.w.trans()) x (a * (1-a));
        delta = (delta * prev.weights.transpose()).multiply_inplace(sigmoid_derivative);
  }
}


#endif // NN_HPP_INCLUDED

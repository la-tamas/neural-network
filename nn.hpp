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

static void write_matrix(std::ofstream& file, const NN_Matrix& m) {
    int rows = m.rows(), cols = m.cols();
    assert(m.data().size() == rows * cols);

    file.write((const char*) &rows, sizeof rows);
    file.write((const char*) &cols, sizeof cols);

    for (matrix_t val : m.data()) {
        file.write((const char*)&val, sizeof val);
    }
}


static NN_Matrix read_matrix(std::ifstream& file) {
    int rows, cols;
    file.read((char*)&rows, sizeof rows);
    file.read((char*)&cols, sizeof cols);
    assert(rows >= 0 && cols >= 0);

    NN_Matrix m(rows, cols);
    std::vector<matrix_t>& data = m.data();
    for (size_t i = 0; i < rows * cols; i++) {
        matrix_t val;
        file.read((char*)&val, sizeof val);
        data[i] = val;
    }

    return m;
}


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

    void save(const char* path) const;
    void load(const char* path);
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

void NN::save(const char* path) const {

  std::ofstream file(path, std::ios::binary);
  assert(!!file);

  file.write((const char*)(&trained), sizeof trained);
  file.write((const char*)(&data_index), sizeof data_index);

  int layer_count = (int) layers.size();
  file.write((const char*)(&layer_count), sizeof layer_count);

  for (const Layer& layer : layers) {
    int activation_count = (int) layer.outputs.cols();

    assert(
      layer.outputs.rows() == 1 &&
      layer.biased.rows() == 1 &&
      activation_count == layer.biased.cols()
    );

    write_matrix(file, layer.biased);
    write_matrix(file, layer.weights);
  }

  file.close();
}


void NN::load(const char* path) {

  // TODO: What if it has layers already.
  //assert(layers.size() == 0 && "Cannot load to an already built nn.");
  layers.clear();

  std::ifstream file(path, std::ios::binary);
  assert(!!file && "Cannot open the nn file.");

  file.read((char*)(&trained), sizeof trained);
  file.read((char*)(&data_index), sizeof data_index);

  int layer_count;
  file.read((char*)&layer_count, sizeof layer_count);
  assert(layer_count >= 0);

  for (int i = 0; i < layer_count; i++) {

    Layer l;

    l.biased = read_matrix(file);
    assert(l.biased.rows() == 1);

    l.outputs.init(1, l.biased.cols());
    l.weights = read_matrix(file);

    layers.push_back(std::move(l));
  }

  // Assert the dimentions are valid.
  for (size_t i = 0; i < layers.size() - 1; i++) {
    const Layer& curr = layers[i];
    const Layer& next = layers[i + 1];
    assert(curr.weights.rows() == curr.outputs.cols());
    assert(curr.weights.cols() == next.outputs.cols());
  }

}


#endif // NN_HPP_INCLUDED

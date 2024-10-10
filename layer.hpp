#pragma once

#ifndef LAYER_HPP_INCLUDED
#define LAYER_HPP_INCLUDED

struct Layer {
    NN_Matrix outputs;
    NN_Matrix biased;
    NN_Matrix weights;

    Layer(int neuron_count = 0);
    Layer(Layer&& other) noexcept;

    Layer next_layer(int neuron_count);

    static void forward(Layer& curr, Layer& prev);
};

Layer::Layer(int neuron_count) {
    outputs.init(1, neuron_count);
    biased.init(1, neuron_count);
}

Layer::Layer(Layer&& other) noexcept :
    outputs(std::move(other.outputs)),
    biased(std::move(other.biased)),
    weights(std::move(other.weights))
{}

Layer Layer::next_layer(int neuron_count) {
  Layer next;
  next.outputs.init(1, neuron_count);
  next.biased.init(1, neuron_count);
  this->weights.init(this->outputs.cols(), next.outputs.cols());
  return next;
}

void Layer::forward(Layer& curr, Layer& prev) {
  curr.outputs = (
    (prev.outputs * prev.weights) += curr.biased
  ).sigmoid();
}

#endif // LAYER_HPP_INCLUDED

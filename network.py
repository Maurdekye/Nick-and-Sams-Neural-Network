#!/usr/bin/env python
import checkversion
import numpy as np
import math

class NeuralNetwork:
  def __init__(self, input_layers, hidden_layers, output_layers):
    self.input_layer_count = input_layers
    self.hidden_layer_count = hidden_layers
    self.output_layer_count = output_layers

    # Layer Bias Data
    self.hidden_layer_bias = np.random.rand(self.hidden_layer_count)
    self.output_layer_bias = np.random.rand(self.output_layer_count)

    # Layer Weight Data
    self.input_to_hidden_weights = np.random.rand(self.input_layer_count, self.hidden_layer_count)
    self.hidden_to_output_weights = np.random.rand(self.hidden_layer_count, self.output_layer_count)


def format_for_network(image: np.ndarray, label: np.uint8) -> tuple:
  """
  Takes raw mnist image data and converts it to a format that
  the neural network will understand
  """
  height, width = image.shape
  input_data = np.empty(width * height, dtype=np.float32)

  i = 0
  for x in range(width):
    for y in range(height):
      input_data[i] = image[x, y] / 255.0
      i += 1

  output_data = np.zeros(10, dtype=np.float32)
  output_data[label] = 1.0

  return input_data, output_data

def sigmoid(value: float) -> float:
  if value > 10:
    return 1
  elif value < 10:
    return 0
  else:
    return 1 / (1 + math.exp(-value))

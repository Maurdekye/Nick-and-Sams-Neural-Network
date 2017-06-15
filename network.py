#!/usr/bin/env python
import checkversion
import numpy as np
import math

class NeuralNetwork:
  def __init__(self, input_layers, hidden_layers, output_layers):
    self.input_layer_size = input_layers
    self.hidden_layer_size = hidden_layers
    self.output_layer_size = output_layers

    # Layer Bias Data
    self.hidden_layer_biases = np.random.rand(self.hidden_layer_size)
    self.output_layer_biases = np.random.rand(self.output_layer_size)

    # Layer Weight Data
    self.input_to_hidden_weights = np.random.rand(self.input_layer_size, self.hidden_layer_size)
    self.hidden_to_output_weights = np.random.rand(self.hidden_layer_size, self.output_layer_size)

  def Evaluate(self, input_data: np.ndarray) -> np.ndarray:
    """
    Computes result based on current network weights and biases
    """
    if input_data.shape[0] != self.input_layer_size:
      raise IndexError(f"Input data length is {input_data.shape[0]}, must match length of input layer size {self.input_layer_size}")

    # Evaulate hidden layer given input values
    hidden_layer_values = np.zeros(self.hidden_layer_size, dtype=np.float32)
    for hidden_node_index in range(self.hidden_layer_size):
      node_value = 0
      for input_node_index in range(self.input_layer_size):
        node_value += input_data[input_node_index] * self.input_to_hidden_weights[input_node_index, hidden_node_index]
      hidden_layer_values[hidden_node_index] = sigmoid(node_value + self.hidden_layer_biases[hidden_node_index])

    # Evaulate output layer given hidden layer values
    output_layer_values = np.zeros(self.output_layer_size, dtype=np.float32)
    for output_node_index in range(self.output_layer_size):
      node_value = 0
      for hidden_node_index in range(self.hidden_layer_size):
        node_value += hidden_layer_values[hidden_node_index] * self.hidden_to_output_weights[hidden_node_index, output_node_index]
      output_layer_values[output_node_index] = sigmoid(node_value + self.output_layer_biases[output_node_index])

    return output_layer_values

  def Cost(self, input_data: list, target_output_data: list):
    """
    Evaluates the cost of the current network from a given set of input and expected output data
    """
    error = 0
    for input_, target_output in zip(input_data, target_output_data):
      generated_output = self.Evaluate(input_)
      for target_output_value, generated_output_value in zip(target_output, generated_output):
        error += (target_output_value - generated_output_value) ** 2
    return error / (2 * len(input_data))

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

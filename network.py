#!/usr/bin/env python
import checkversion
import random
import numpy as np
import math
import json
from os import path

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

    # Gradient Vectors
    self.hidden_biases_gradient = np.zeros_like(self.hidden_layer_biases)
    self.output_biases_gradient = np.zeros_like(self.output_layer_biases)

    self.input_to_hidden_weights_gradient = np.zeros_like(self.input_to_hidden_weights)
    self.hidden_to_output_weights_gradient = np.zeros_like(self.hidden_to_output_weights)

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

  def ComputeGradients(self, input_data: list, target_output_data: list):
    """
    Naively computes the gradient vectors for the current weights and biases using the simple numerical
    definition of the derivative, lim h>0 (f(x + h) - f(x)) / h
    """
    delta = 1e-6
    normal_cost = self.Cost(input_data, target_output_data)

    # Evaluate Gradient for Hidden Layer Biases
    for i in range(self.hidden_layer_biases.shape[0]):
      original_bias_value = self.hidden_layer_biases[i]
      self.hidden_layer_biases[i] += delta
      plusdelta_cost = self.Cost(input_data, target_output_data)
      self.hidden_layer_biases[i] = original_bias_value
      self.hidden_biases_gradient[i] = (plusdelta_cost - normal_cost) / delta

    # Evaluate Gradient for Output Layer Biases
    for i in range(self.output_layer_biases.shape[0]):
      original_bias_value = self.output_layer_biases[i]
      self.output_layer_biases[i] += delta
      plusdelta_cost = self.Cost(input_data, target_output_data)
      self.output_layer_biases[i] = original_bias_value
      self.output_biases_gradient[i] = (plusdelta_cost - normal_cost) / delta

    # Evaluate Gradient for Input Layer to Hidden Layer Weights
    for i in range(self.input_to_hidden_weights.shape[0]):
      for h in range(self.input_to_hidden_weights.shape[1]):
        original_bias_value = self.input_to_hidden_weights[i, h]
        self.input_to_hidden_weights[i, h] += delta
        plusdelta_cost = self.Cost(input_data, target_output_data)
        self.input_to_hidden_weights[i, h] = original_bias_value
        self.input_to_hidden_weights_gradient[i, h] = (plusdelta_cost - normal_cost) / delta

    # Evaluate Gradient for Input Layer to Hidden Layer Weights
    for h in range(self.hidden_to_output_weights.shape[0]):
      for o in range(self.hidden_to_output_weights.shape[1]):
        original_bias_value = self.hidden_to_output_weights[h, o]
        self.hidden_to_output_weights[h, o] += delta
        plusdelta_cost = self.Cost(input_data, target_output_data)
        self.hidden_to_output_weights[h, o] = original_bias_value
        self.hidden_to_output_weights_gradient[h, o] = (plusdelta_cost - normal_cost) / delta

  def Train(self, input_data: list, target_output_data: list, learning_rate: float) -> float:
    """
    Applies gradient vectors by stepping in the negative gradient of the cost function.
    """
    ComputeGradients(input_data, target_output_data)
    self.hidden_layer_biases -= learning_rate * self.hidden_biases_gradient
    self.output_layer_biases -= learning_rate * self.output_biases_gradient
    self.input_to_hidden_weights -= learning_rate * self.input_to_hidden_weights_gradient
    self.hidden_to_output_weights -= learning_rate * self.hidden_to_output_weights_gradient

  def Save(self, filename: str):
    """
    Saves current weights and biases to a file so they can be loaded for later use
    """
    data_object = {
      "input_layer_count" : self.input_layer_size,
      "hidden_layer_count" : self.hidden_layer_size,
      "output_layer_count" : self.output_layer_size,

      "hidden_layer_biases" : self.hidden_layer_biases.tolist(),
      "output_layer_biases" : self.output_layer_biases.tolist(),

      "input_to_hidden_weights" : self.input_to_hidden_weights.tolist(),
      "hidden_to_output_weights" : self.hidden_to_output_weights.tolist()
    }

    with open(filename, "w") as f:
      json.dump(data_object, f)

  def Load(self, filename: str) -> bool:
    try:
      with open(filename, "r") as f:
        data_object = json.loads(f)

        self.input_layer_count = data_object["input_layer_count"]
        self.hidden_layer_count = data_object["hidden_layer_count"]
        self.output_layer_size = data_object["output_layer_size"]

        self.hidden_layer_biases = np.array(data_object["hidden_layer_biases"])
        self.output_layer_biases = np.array(data_object["output_layer_biases"])

        self.input_to_hidden_weights = np.array(data_object["input_to_hidden_weights"])
        self.hidden_to_output_weights = np.array(data_object["hidden_to_output_weights"])

      return True
    except:
      return False

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

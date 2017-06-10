#!/usr/bin/env python
import checkversion
import numpy as np

class NeuralNetwork:
  def __init__(self, input_layers, hidden_layers, output_layers):
    self.input_layers = input_layers
    self.hidden_layers = hidden_layers
    self.output_layers = output_layers

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
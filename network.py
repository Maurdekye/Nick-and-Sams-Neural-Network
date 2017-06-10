import numpy as np

class NeuralNetwork:
  def __init__(self, input_layers, hidden_layers, output_layers):
    pass

def format_for_network(image: list, label: bytes) -> list:
    height, width = image.shape
    input_data = np.zeros(width * height, dtype=np.float32)
    output_data = np.zeros(10, dtype=np.float32)

    i = 0
    for x in range(width):
        for y in range(height):
            input_data[index] = image[x, y] / 255.0
            index += 1

    output_data[label] = 1.0

    return input_data, output_data
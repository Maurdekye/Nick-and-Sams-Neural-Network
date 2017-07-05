#!/usr/bin/env python
import checkversion
import numpy as np

class MNISTFormatException(Exception):
  pass

def bytes_to_int(bytestring: bytes) -> int:
  return int.from_bytes(bytestring, byteorder="big")

def load(imagefile_path: str, labelfile_path: str, data_limit=None) -> list:
  """
  Loads MNIST data into a list containing tuples with two items each
  The first argument is a 2d numpy ndarray of bytes containing the image data of a digit
  the second argument is a single byte which is the number the image data ought to represent
  the third argument is an optional limit to only load a few items
  """
  with open(imagefile_path, "rb") as image_file:
    with open(labelfile_path, "rb") as label_file:

      imagefile_designator = bytes_to_int(image_file.read(4))
      if imagefile_designator != 2051:
        raise MNISTFormatException(f"Image file {imagefile_path} not formatted properly")

      labelfile_designator = bytes_to_int(label_file.read(4))
      if labelfile_designator != 2049:
        raise MNISTFormatException(f"Label file {labelfile_path} not formatted properly")

      num_items = bytes_to_int(image_file.read(4))
      num_items_labelfile = bytes_to_int(label_file.read(4))
      if num_items != num_items_labelfile:
        raise MNISTFormatException(f"Image file contains {num_items} elements while Label file contains {num_items_labelfile} elements: they must be the same")

      image_height = bytes_to_int(image_file.read(4))
      image_width = bytes_to_int(image_file.read(4))

      image_data = []

      if data_limit != None:
        num_items = data_limit

      for i in range(num_items):
        image = np.empty((image_height, image_width), dtype=np.uint8)
        label = np.uint8(bytes_to_int(label_file.read(1)))
        for y in range(image_height):
          for x in range(image_width):
            image[y, x] = bytes_to_int(image_file.read(1))
        image_data.append((image, label))

      return image_data

def image_string(image_bytes: np.ndarray, display_threshold=127) -> str:
  final_string = ""
  for y in range(image_bytes.shape[0]):
    for x in range(image_bytes.shape[1]):
      value = " "
      if image_bytes[y, x] > display_threshold:
        value = "0"
      final_string += value + " "
    final_string += "\n"
  return final_string

def generate_batch(dataset: list, batch_size: int) -> list:
  items = random.sample(range(len(dataset)), batch_size)
  return [dataset[i] for i in items]
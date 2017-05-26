#!/usr/bin/env python
import sys
major_version, minor_version = sys.version_info[:2]
if major_version < 3 or minor_version < 6:
  raise Exception("Python version must be at least 3.6 to run")

import numpy as np

class MNISTFormatException(Exception):
  pass

def bytes_to_int(bytestring: bytes) -> int:
  return int.from_bytes(bytestring, byteorder="big")

def load(imagefile_path: str, labelfile_path: str) -> list:
  """
  Loads MNIST data into a list containing tuples with two items each
  The first item is a 2d numpy ndarray of bytes containing the image data of a digit
  the second item is a single byte which is the number the image data ought to represent
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

      for i in range(num_items):
        image = np.empty((image_height, image_width), dtype=np.uint8)
        label = np.uint8(bytes_to_int(label_file.read(1)))
        for y in range(image_height):
          for x in range(image_width):
            image[y, x] = bytes_to_int(image_file.read(1))
        image_data.append((image, label))

      return image_data
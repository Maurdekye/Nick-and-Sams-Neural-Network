#!/usr/bin/env python
import checkversion
import mnist

training_data = mnist.load("train-images.idx3-ubyte", "train-labels.idx1-ubyte")
testing_data = mnist.load("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte")
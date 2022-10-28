import numpy as np
import math

NORMAL_MEAN = 0
NORMAL_STD = 1


class NeuralNetwork():

    def __init__(self, layer_sizes):

        self.biases = [np.random.normal(NORMAL_MEAN, NORMAL_STD, size=(x, 1))
                       for x in layer_sizes[1:]]
        self.weights = [np.random.normal(NORMAL_MEAN, NORMAL_STD, size=(y, x))
                        for x, y in zip(layer_sizes[: -1], layer_sizes[1:])]
        self.layer_sizes = layer_sizes

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def relu(self, x):
        return x if x > 0 else 0

    def forward(self, x):
        start = x.reshape(self.layer_sizes[0], 1)

        z1 = self.biases[0] + np.dot(self.weights[0], start)
        a1 = np.vectorize(np.tanh)(z1)

        z2 = self.biases[1] + np.dot(self.weights[1], a1)
        a2 = np.vectorize(self.sigmoid)(z2)

        return a2

import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, x, y):
        self.input      = x
        self.weights1   = np.random.rand(self.input.shape[1],4) 
        self.weights2   = np.random.rand(4,1)                 
        self.y          = y
        self.output     = np.zeros(self.y.shape)

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))


# Input
# [ value_theta, value_and, value_and ]
input_values = [[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]
input_values = np.asarray(input_values)

# Output
output_values = [0 , 0, 0, 1]
output_values = np.asarray(output_values)

NN = NeuralNetwork(input_values, output_values)
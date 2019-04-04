import numpy as np
from NeuralNetwork import NeuralNetwork 

#Outputs and Inputs
# X = np.asarray([ [-1, 1, -1, 1], [-1, -1, 1, 1] ])
# Yd = np.asarray([[-1, 1, 1, -1]])
# Se definen las entradas de los 16 casos de entrenamiento.
X = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],  # Variable x.
    [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],  # Variable y.
    [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],  # Variable w.
    [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]]) # Variable z.

# Se calculan las salidas esperadas.
Yd = []
# Para cada caso de entrenamiento...
for i in range(X.shape[1]):
    # Se obtienen las 4 entradas del caso.
    w, x, y, z = np.array(X[:, i], dtype=bool)
    # Se obtienen la salida esperada, ejecutando la función boleana.
    Yd.append(x and not(y) or (y or not(w)) or not(not(z) or w))

Yd = np.array([Yd], dtype=int)
print("Y")
print(Yd)
print("\n")
#Layers
L = [4, 5, 1]

#Data network
eta = 0.2
alfa = 0
epsilon = 0.0001

neural_network = NeuralNetwork()
neural_network.Training(X, Yd, L, eta, alfa, epsilon, 10000)
neural_network.CreateChartError(neural_network.list_mse, neural_network.list_period)
neural_network.PrintRecordNeuralNetwork()
neural_network.PrintWeightNeuralNetwork()
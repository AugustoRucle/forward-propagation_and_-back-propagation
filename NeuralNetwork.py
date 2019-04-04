import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable

class NeuralNetwork:

    def __init__(self):
        self.list_mse = []
        self.list_period = []
        self.final_weight  = []
        self.layers = []
        self.periods = 1
        self.mses = 1
        self.final_records = []

    #X, Yd, L, eta, alfa, epsilon, epoca, mse
    def Training(self, array_input, array_output, layers, eta, alfa, epsilon, MAX_period):
        self.layers = layers
        size_array_input, amount_element_input = len(array_input), len(array_input[0])    
        size_array_output, amount_element_output = len(array_output), len(array_output[0])
        amount_layers = len(layers)

        array_weight = self.InitWeigthLayers(amount_layers, layers)
        array_delta_weight = self.InitDeltaWeight(amount_layers, array_weight) 
        array_V = self.InitArrayV(amount_layers, layers)
        array_Y = self.InitArrayY(amount_layers, layers)

        while ((self.mses > epsilon) and (self.periods <= MAX_period) ):
            e, err, self.periods = np.zeros((size_array_output, amount_element_input)), np.zeros((1, amount_element_input)), self.periods + 1
            records = []

            for j in range(amount_element_input):
                array_V[0][0:-1] = np.vstack(array_input[:, j])

                #Forward-propagation
                for f in range(amount_layers-1):
                    array_Y[f] = np.dot(array_weight[f], array_V[f])

                    if(f < amount_layers - 2):
                        array_V[f+1][0:-1] = self.Activation(array_Y[f])
                    else:
                        array_V[f+1] = self.Activation(array_Y[f])

                #Singal Error
                e[:,j] = array_output[0][j] - array_V[-1]

                #Saved record
                records.append((j + 1, array_input[:, j], array_V[-1], array_output[0][j], e[:,j] ))

                #Energy error
                if(len(array_output) == 1):
                    #output scale
                    err[0][j] = 0.5 * (e[:, j]**2)

                elif(len(array_output) > 1):
                    #output vector
                    err[0][j] = 0.5 * np.sum((e[:, j]**2))
                
                delta = e[:,j] * self.ActivationInversa(array_Y[-1])

                #Back-propagation
                for i in range(amount_layers-1):
                    index = amount_layers-(i+2)

                    array_delta_weight[index] = eta * delta * array_V[index].transpose() + (alfa * array_delta_weight[index])

                    array_weight[index] = array_weight[index] + array_delta_weight[index]

                    if((amount_layers-(i+1)) > 1):
                        delta = self.ActivationInversa(array_Y[index-1]) * (( delta.transpose() * array_weight[index][:, 0:-1] ).transpose())
            
            self.final_weight = array_weight.copy()
            self.final_records = records.copy()
            records.clear()
            self.mses = (1/amount_element_input) * np.sum(err)
            self.list_mse.append(self.mses)
            self.list_period.append(self.periods)
        
    def CreateChartError(self, list_mse, list_period):
        plt.title('Evolución del error cuadratico medio (mse)')
        plt.xlabel('Generaciones')
        plt.ylabel('Valor de mse')
        plt.plot(list_period, list_mse)
        plt.show()  

    def PrintRecordNeuralNetwork(self):
        table = PrettyTable()
        table.field_names = ["Iteracion", "Entradas", "R-Obtenidos", "R-Deseados", "Error"]
        for row in self.final_records:
            table.add_row(row)
        self.PrintTable(table)

    def PrintWeightNeuralNetwork(self):
        table = PrettyTable()
        table.field_names = ["# Capa","# Perceptron", "# Peso", "Pesos"]
        amount_elements, number_weight = len(self.final_weight), 0
        for i in range(amount_elements):
            for j in range(len(self.final_weight[i])):
                for z in range(len(self.final_weight[i][j])):
                    table.add_row((i+1, j+1, number_weight,  self.final_weight[i][j][z]))
                    number_weight += 1
        self.PrintTable(table)
        
    def PrintTable(self, table):
        print("\n\n\n\n")
        print(table)

    def Activation(self, array_y):
        return np.tanh(array_y) 

    def ActivationInversa(self, array_y):
        return 1.0 - (np.tanh(array_y) ** 2)

    def InitWeigthLayers(self, amount_layers, layers):
        array_weight = []
        for i in range(amount_layers - 2):
            array_weight.append(-1+2*(np.random.rand(layers[i+1], layers[i]+1)))
        array_weight.append(-1+2*(np.random.rand(layers[-1], layers[-2]+1)))
        return array_weight
# 
    def InitDeltaWeight(self, amount_layers, array_weight):
        array_delta_weight = []
        for i in range(amount_layers - 1):
            array_delta_weight.append(np.zeros(array_weight[i].shape))
        return array_delta_weight

    def InitArrayV(self, amount_layers, layers):
        array_V = []
        for i in range(amount_layers-1):
            array = np.zeros(layers[i]+1)
            array[-1] = 1
            array_V.append(np.vstack(array)) 
        array_V.append(np.vstack(np.zeros(layers[-1])))
        return array_V

    def InitArrayY(self, amount_layers, layers):
        array_Y = []
        for i in range(amount_layers-2):
            array_Y.append(np.vstack(np.zeros(layers[i]+1))) 
        array_Y.append(np.vstack(np.zeros(layers[-1])))
        return array_Y


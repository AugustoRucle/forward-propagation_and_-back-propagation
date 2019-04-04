################################
#Librerias                      #
################################

import numpy as np
import matplotlib.pyplot as plt

################################
#Funciones                      #
################################
def main():
    
    

    X = np.asarray([ [-1, 1, -1, 1], [-1, -1, 1, 1] ])
    Yd = np.asarray([[-1, 1, 1, -1]])

    #Layers
    L = [2, 3, 1]

    #Data network
    eta = 0.2
    alfa = 0
    epsilon = 0.0001
    
    Retropropagamacion(X, Yd, L, eta, alfa, epsilon)

def Retropropagamacion(X, Yd, L, eta, alfa, epsilon):
    epoca = 1 
    mse = 1

    list_mse = []
    list_epoca = []

    #[amount of Y, amount of X]   
    x, p = len(X), len(X[0])    
    q, pd = len(Yd), len(Yd[0])
    amountLayers = len(L)

    array_weight = initWeigthLayers(amountLayers, L)
    array_DeltaWeight = initDeltaWeight(amountLayers, array_weight)
    array_V = initArrayV(amountLayers, L)
    array_Y = initArrayY(amountLayers, L)

    while ((mse > epsilon) and (epoca <= 10000) ):
        e, err, epoca = np.zeros((q, p)), np.zeros((1, p)), epoca + 1

        for j in range(p):
            array_V[0][0:-1] = np.vstack(X[:, j])

            #Forward-propagation
            for f in range(amountLayers-1):
                array_Y[f] = np.dot(array_weight[f], array_V[f])

                if(f < amountLayers - 2):
                    array_V[f+1][0:-1] = activacion(array_Y[f])
                else:
                    array_V[f+1] = activacion(array_Y[f])

            #Singal Error
            e[:,j] = Yd[0][j] - array_V[-1]

            #Energy error
            if(len(Yd) == 1):
                #output scale
                err[0][j] = 0.5 * (e[:, j]**2)

            elif(len(Yd) > 1):
                #output vector
                err[0][j] = 0.5 * np.sum((e[:, j]**2))
            
            delta = e[:,j] * activacionInversa(array_Y[-1])

            #Back-propagation

            for i in range(amountLayers-1):
                index = amountLayers-(i+2)

                array_DeltaWeight[index] = eta * delta * array_V[index].transpose() + (alfa * array_DeltaWeight[index])

                array_weight[index] = array_weight[index] + array_DeltaWeight[index]

                if((amountLayers-(i+1)) > 1):
                    delta = activacionInversa(array_Y[index-1]) * (( delta.transpose() * array_weight[index][:, 0:-1] ).transpose())

        
        mse = (1/p) * np.sum(err)
        list_mse.append(mse)
        list_epoca.append(epoca)

    createGrafic(list_mse, list_epoca)

def createGrafic(list_mse, list_epoca):
    plt.plot(list_epoca, list_mse)
    plt.show()  

def activacion(array_y):
    return np.tanh(array_y) 

def activacionInversa(array_y):
    return 1.0 - (np.tanh(array_y) ** 2)

def initWeigthLayers(amountlayers, L):
    array_weight = []
    for i in range(amountlayers - 2):
        array_weight.append(-1+2*(np.random.rand(L[i+1], L[i]+1)))
    array_weight.append(-1+2*(np.random.rand(L[-1], L[-2]+1)))
    return array_weight

def initDeltaWeight(amount, array_weight):
    array_DeltaWeight = []
    for i in range(amount - 1):
        array_DeltaWeight.append(np.zeros(array_weight[i].shape))
    return array_DeltaWeight

def initArrayV(amount, L):
    array_V = []
    for i in range(amount-1):
        array = np.zeros(L[i]+1)
        array[-1] = 1
        array_V.append(np.vstack(array)) 
    array_V.append(np.vstack(np.zeros(L[-1])))
    return array_V

def initArrayY(amount, L):
    array_Y = []
    for i in range(amount-2):
        array = np.zeros(L[i]+1)
        array_Y.append(np.vstack(array)) 
    array_Y.append(np.vstack(np.zeros(L[-1])))
    return array_Y

################################
#Start algorithm                #
################################
main()














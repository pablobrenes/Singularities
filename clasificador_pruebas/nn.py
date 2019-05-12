import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd


def loadData(name):
    data = pd.read_csv(name)

    # Obtener los datos como un tensor
    dataFrame = pd.DataFrame(data=data)
    raw = torch.tensor(dataFrame.values)

    # Barajar los datos
    raw = raw[torch.randperm(raw.size()[0])]

    # Obtener las entradas, x. Primeras 7 columnas
    x = raw[:, :-1]

    # Obtener las etiquetas, y. Última columna
    y = raw[:, -1]

    # Convertir las etiquetas a una versión "one hot vector"
    # [0, 1, 2] -> [[1, 0, 0], [0, 1, 0] [0, 0, 1]]
    posiblesVectores = torch.eye(10)
    y = posiblesVectores[y]

    return x.float(), y.float()


# Datos
x, y = loadData("../led.csv")
cantidadDatos = x.size()[0]
cantidadEntrenamiento = int(cantidadDatos * 0.7)

# x.cuda()
# y.cuda()

# 70% para entrenammiento, 30% para prueba
xTest = x[cantidadEntrenamiento:]
yTest = y[cantidadEntrenamiento:]

x = x[:cantidadEntrenamiento]
y = y[:cantidadEntrenamiento]

# Perceptrón multicapa para clasificación.
# 7 neuronas de entrada, para cada led.
# 10 neuronas de salida, para cada salida esperada de los datos.
# Una capa oculta de 10 neuronas.
# La entrada se conecta a la oculta con la tangente hiperbólica
# la oculta se conecta a la salida con softmax para obtener diez salidas.

# Capas
inputLayerSize, hiddenLayerSize, outputLayerSize = 7, 14, 10

# Modelo
modelo = nn.Sequential(
    nn.Linear(inputLayerSize, hiddenLayerSize),
    nn.ReLU(),
    nn.Linear(hiddenLayerSize, outputLayerSize),
    nn.Softmax(1))

# Función de error
errorF = nn.MSELoss()

# Optimizador, decenso del gradiente
aprendizaje = 0.001
optimizador = optim.SGD(modelo.parameters(), aprendizaje)

# Entrenamiento
epochs = 4001

for epoch in range(epochs):
    salida = modelo(x)

    error = errorF(salida, y)

    if not (epoch % 500):
        print("Epoch:", epoch)
        print("Error training", error.item())

        salidaTest = modelo(xTest)
        errorTest = errorF(salidaTest, yTest)
        print("Error test", errorTest.item())
        print()

    # Reiniciar el cálculo del gradiente
    optimizador.zero_grad()

    # Backpropagation, obtener los gradientes
    error.backward()

    # Optmizar
    optimizador.step()
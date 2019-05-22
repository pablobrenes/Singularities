import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd


# Obtener los valores de entrada y etiquetas del dataset
def load_data(name):
    data = pd.read_csv(name)

    # Obtener los datos como un tensor
    dataFrame = pd.DataFrame(data=data)
    raw = torch.tensor(dataFrame.values)

    # Barajar los datos
    raw = raw[torch.randperm(raw.size()[0])]

    # Obtener las entradas, x. Primeras 7 columnas
    x = raw[:, :7]

    # Obtener las etiquetas, y. Última columna
    y = raw[:, 7]

    # Convertir las etiquetas a una versión "one hot vector"
    # [0, 1, 2] -> [[1, 0, 0], [0, 1, 0] [0, 0, 1]]
    posiblesVectores = torch.eye(10)
    y = posiblesVectores[y]

    return x.float(), y


def get_proportion(value, target, proportion):
    cantidad_datos = value.size()[0]
    cantidad_entrenamiento = int(cantidad_datos * proportion)

    # Obtener solo la proporción necesaria
    value_test = value[cantidad_entrenamiento:]
    target_test = target[cantidad_entrenamiento:]

    value_train = value[:cantidad_entrenamiento]
    target_train = target[:cantidad_entrenamiento]

    return value_train, target_train, value_test, target_test


# Perceptrón multicapa para clasificación.
# 7 neuronas de entrada, para cada led.
# 10 neuronas de salida, para cada salida esperada de los datos.
# Una capa oculta de 10 neuronas.
# La entrada se conecta a la oculta con la tangente hiperbólica
# la oculta se conecta a la salida con softmax para obtener diez salidas.
def create_model(inputLayerSize=7, hiddenLayerSize=10, outputLayerSize=10):
    # Base model
    model = nn.Sequential(
        nn.Linear(inputLayerSize, hiddenLayerSize),
        nn.Tanh(),
        nn.Linear(hiddenLayerSize, outputLayerSize),
        nn.Softmax(1))
    return model


def train_model(model, input, target, input_test, target_test):
    # Función de error
    error_function = nn.BCELoss()

    # Definir el método de optimización y la velocidad de aprendizaje
    learning_rate = 0.01
    optimizer = optim.Adagrad(model.parameters(), learning_rate)

    # Ciclos necesarios para entrenar
    epochs = 2001

    for epoch in range(epochs):
        # Salida del modelo
        out = model(input)

        # Error del modelo
        error = error_function(out, target)

        # Reiniciar el cálculo del gradiente
        optimizer.zero_grad()

        # Propagación del error
        error.backward()

        # Optimizar los parámetros
        optimizer.step()

    return model, error


def test_model(model, input, target):
    out = model(input)

    _, estimated_class = torch.max(out, 1)
    _, real_class = torch.max(target, 1)

    matches = (estimated_class == real_class).float()

    return torch.mean(matches)


def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(path):
    model = create_model()
    model.load_state_dict(torch.load(path))
    return model


def main():
    value, target = load_data("../led.csv")

    avgs = []
    errors = []

    for i in range(50):
        value_train, target_train, value_test, target_test = get_proportion(value, target, .7)
        model = create_model()

        model, error = train_model(model, value_train, target_train, value_test, target_test)
        avg_hits = test_model(model, value_test, target_test)

        avgs.append(avg_hits.item())
        errors.append(error.item())
        print("Train:", i)

    avgs = torch.tensor(avgs)
    errors = torch.tensor(errors)

    ma, sa = torch.mean(avgs), torch.std(avgs)
    me, se = torch.mean(errors), torch.std(errors)

    print("Avg", ma, sa)
    print("Std", me, se)


    # save_model(model, 'test.model')
    #
    # model_ = load_model('test.model')
    #
    # avg_hits = test_model(model_, value_test, target_test)
    # print("Porcentaje de datos correctamente clasificados:", avg_hits)

    """
    Avg tensor(0.7517) tensor(0.0035)
    Std tensor(0.1387) tensor(0.0011)
    """


if __name__ == '__main__':
    main()

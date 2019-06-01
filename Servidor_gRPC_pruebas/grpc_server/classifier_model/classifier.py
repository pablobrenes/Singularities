import torch
import torch.optim as optim
import torch.nn as nn


class Classifier(object):

    def __init__(self, inputLayerSize=7, hiddenLayerSize=21, outputLayerSize=10):
        self.model = nn.Sequential(
            nn.Linear(inputLayerSize, hiddenLayerSize),
            nn.Tanh(),
            nn.Linear(hiddenLayerSize, outputLayerSize),
            nn.Softmax(1))

    def train_model(self, values, targets):
        # Función de error
        error_function = nn.CrossEntropyLoss()

        # Definir el método de optimización y la velocidad de aprendizaje
        learning_rate = 0.005
        optimizer = optim.Adagrad(self.model.parameters(), learning_rate)

        # Ciclos necesarios para entrenar
        epochs = 4000

        for epoch in range(epochs):
            # Salida del modelo
            out = self.model(values)

            # Error del modelo
            error = error_function(out, targets)

            # Reiniciar el cálculo del gradiente
            optimizer.zero_grad()

            # Propagación del error
            error.backward()

            # Optimizar los parámetros
            optimizer.step()

    def consult_model(self, input):
        value = torch.tensor([input]).float()
        out = self.model(value)

        _, estimated_class = torch.max(out, 1)

        return estimated_class.item()

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))

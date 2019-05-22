import torch
import pandas as pd


class DataLoader(object):

    def __init__(self, path):
        self.path = path

    def get_data(self, proportion):
        values, targets = self.__load_data()

        cantidad_datos = values.size()[0]
        cantidad_entrenamiento = int(cantidad_datos * proportion)

        # Obtener solo la proporción necesaria
        value_train = values[:cantidad_entrenamiento]
        target_train = targets[:cantidad_entrenamiento]

        return value_train, target_train

    def __load_data(self):
        # Leer los valores desde el archivo csv.
        data = pd.read_csv(self.path)

        # Obtener los datos como un tensor
        data_frame = pd.DataFrame(data=data)
        raw_data = torch.tensor(data_frame.values)

        # Barajar los datos
        raw_data = raw_data[torch.randperm(raw_data.size()[0])]

        # Obtener las entradas, values. Primeras 7 columnas
        values = raw_data[:, :7]

        # Obtener las etiquetas, targets. Última columna
        targets = raw_data[:, 7]

        # Convertir las etiquetas a una versión "one hot vector"
        # [0, 1, 2] -> [[1, 0, 0], [0, 1, 0] [0, 0, 1]]
        posibles_vectores = torch.eye(10)
        targets = posibles_vectores[targets]

        return values.float(), targets

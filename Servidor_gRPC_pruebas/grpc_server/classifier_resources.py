import json
from datetime import datetime as dt


class ModelEntity(object):

    def __init__(self, name, proportion, url, file_location, datetime=dt.now(), trained=False):
        self.name = name
        self.proportion = proportion
        self.url = url
        self.file_location = file_location
        if type(datetime) is str:
            self.datetime = dt.strptime(datetime, "%Y-%m-%d %H:%M:%S.%f")
        else:
            self.datetime = datetime
        self.trained = trained

    def get_dict(self):
        dictionary = {
            "name": self.name,
            "proportion": self.proportion,
            "url": self.url,
            "file_location": self.file_location,
            "datetime" : str(self.datetime),
            "trained": self.trained
        }
        return dictionary

    def __str__(self):
        string = \
            f"Nombre del modelo: {self.name}\n" \
            f"Proporción a usar: {self.proportion}\n" \
            f"Fecha de creación: {self.datetime}"
        return string


def read_model_entity_database(database_file_name):
    model_entity_list = []

    with open(database_file_name) as database_file:
        for item in json.load(database_file):
            model_entity = ModelEntity(
                item["name"], item["proportion"], item["url"], item["file_location"], item["datetime"], item["trained"]
            )
            model_entity_list.append(model_entity)

    return model_entity_list


def write_model_entity_database(database_file_name, model_entity_list):
    with open(database_file_name, "w+") as database_file:
        model_entity_list_dict = [model_entity.get_dict() for model_entity in model_entity_list]
        _json = json.dumps(model_entity_list_dict, indent=4)
        database_file.write(_json)

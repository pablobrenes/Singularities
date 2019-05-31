from threading import Thread, Lock
from concurrent import futures
from queue import Queue
import logging
import time
import uuid
import os

import grpc
import requests

import classifier_resources
from grpc_modules import classifier_pb2
from grpc_modules import classifier_pb2_grpc
from classifier_model.classifier import Classifier
from classifier_model.data_loader import DataLoader

ONE_DAY_IN_SECONDS = 24 * 60 * 60
AMOUNT_WORKERS = 5
MAIN_DIR = "server_data"
TEMP_FILES_DIR = os.path.join(MAIN_DIR, "temp_datasets_storage")
MODELS_DIR = os.path.join(MAIN_DIR, "models")
DATABASE_FILE_NAME = os.path.join(MAIN_DIR, "model_entity_database.json")


def get_model_entity(model_entyty_database, name):
    for model_entity in model_entyty_database:
        if model_entity.name == name:
            return model_entity
    return None


def add_model_entity(model_entyty_database, model_entity):
    model_entyty_database.append(model_entity)


def update_trained_value(model_entyty_database, name):
    for model_entyty in model_entyty_database:
        if model_entyty.name == name:
            model_entyty.trained = True
            return True
    return False


def load_model(file_location):
    # Obtener un clasificador
    classifier = Classifier()

    # Cargar el clasificador
    classifier.load_model(file_location)

    return classifier


class ClassifierServicer(classifier_pb2_grpc.ClassifierServicer):

    def __init__(self):
        self.model_entyty_database = classifier_resources.read_model_entity_database(DATABASE_FILE_NAME)
        # Cola de trabajo
        self.work_queue = Queue()

    def CreateModel(self, request, context):
        logging.info('Request for create a model')
        # Obtener un nombre de archivo para el modelo
        filename = request.name.replace(" ", "_") + '_' + str(uuid.uuid1())
        file_location = os.path.join(MODELS_DIR, filename)

        # Obtener una representación de la entidad para la base de datos.
        model_entity = classifier_resources.ModelEntity(
            request.name, request.proportion, request.url, file_location
        )

        # Agregar la representación a la cola de trabajo para su entrenamiento
        self.work_queue.put_nowait(model_entity)

        # Guardar la representación
        add_model_entity(self.model_entyty_database, model_entity)

        return classifier_pb2.ModelRepresentation(detail=str(model_entity))

    def ConsultModel(self, request, context):
        logging.info('Request for consult a model')
        # Obtener la representación del modelo
        model_entity = get_model_entity(self.model_entyty_database, request.name)

        # Si el modelo existe
        if model_entity:
            # Si el modelo ya está entrenado
            if model_entity.trained:
                classifier = load_model(model_entity.file_location)
                output = classifier.consult_model(request.leds)
                return classifier_pb2.ConsultModelResponse(output=output)
            else:
                pass
        else:
            # El modelo no existe
            return classifier_pb2.ConsultModelResponse(output=10)


def download_file(url, filename):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=4096):
                if chunk:
                    f.write(chunk)


def train_model_worker(work_queue, model_entyty_database):
    while True:
        # Obtener el elemento de la cola a procesar.
        model_entity = work_queue.get()
        logging.info('Found a job')

        # Descargar el dataset desde el url.
        download_filename = os.path.join(TEMP_FILES_DIR, str(uuid.uuid1()))
        download_file(model_entity.url, download_filename)
        logging.info('Dataset downloaded')

        # Crear un model no entrenado.
        classifier = Classifier()

        # Comenzar el entrenamiento del modelo.
        logging.info('Model began to train')
        data_loader = DataLoader(download_filename)
        values, targets = data_loader.get_data(model_entity.proportion)
        classifier.train_model(values, targets)
        logging.info('Model trained')

        # Eliminar el archivo temporal del dataset
        os.remove(download_filename)

        # Cuando el entrenamiendo esté completo, guardar el modelo en memoria.
        classifier.save_model(model_entity.file_location)

        # Actualizar el valor de entrenamiento en la base de datos del servidor y escribir el archivo.
        lock = Lock()
        lock.acquire()
        update_trained_value(model_entyty_database, model_entity.name)
        classifier_resources.write_model_entity_database(DATABASE_FILE_NAME, model_entyty_database)
        lock.release()
        logging.info('Model updated in db')

        # Indicar a la cola que el elemento fue procesado
        work_queue.task_done()
        logging.info('Job done')


def start_train_model_workers(amount_workers, task, work_queue, model_entyty_database):
    for _ in range(amount_workers):
        worker = Thread(target=task, args=(work_queue, model_entyty_database,))
        worker.setDaemon(True)
        worker.start()


def verify_files():
    if not os.path.isdir(MAIN_DIR):
        os.mkdir(MAIN_DIR)
    if not os.path.isdir(MODELS_DIR):
        os.mkdir(MODELS_DIR)
    if not os.path.isdir(TEMP_FILES_DIR):
        os.mkdir(TEMP_FILES_DIR)
    if not os.path.isfile(DATABASE_FILE_NAME):
        with open(DATABASE_FILE_NAME, "w+") as f:
            f.write('[]\n')
            f.close()


def serve():
    # Verificar integridad de la estructura de archivos necesaria
    verify_files()

    classifier_server = grpc.server(futures.ThreadPoolExecutor())
    classifier_servicer = ClassifierServicer()
    classifier_pb2_grpc.add_ClassifierServicer_to_server(classifier_servicer, classifier_server)

    # Iniciar los trabajadores que atienden la cola de trabajos para entrenar modelos
    # Iniciar trabajadores
    start_train_model_workers(AMOUNT_WORKERS, train_model_worker, classifier_servicer.work_queue,
                              classifier_servicer.model_entyty_database)

    classifier_server.add_insecure_port('[::]:50051')
    classifier_server.start()

    try:
        while True:
            time.sleep(ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        # Esperar a que la cola de trabajo termine
        classifier_servicer.work_queue.join()
        classifier_server.stop(0)


def run_server():
    logging.basicConfig(level=logging.INFO)
    serve()

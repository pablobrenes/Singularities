from __future__ import print_function
import logging
import sys

import grpc

from grpc_modules import classifier_pb2
from grpc_modules import classifier_pb2_grpc

URL = 'https://storage.googleapis.com/globalimages/led.csv'


def consult_model(stub, name, leds):
    consult_model_request = classifier_pb2.ConsultModelRequest(name=name)
    consult_model_request.leds.extend(leds)

    response = stub.ConsultModel(consult_model_request)

    print(f'Salida del modelo: {response.output}')


def create_model(stub, _name, _proportion, _url):
    create_model_request = classifier_pb2.CreateModelRequest(
        name=_name, proportion=_proportion, url=_url
    )

    response = stub.CreateModel(create_model_request)

    print(f'Detalle del modelo:\n{response.detail}')


def menu():
    print('Seleccione una opción para continuar.')
    print('\t1) Crear un nuevo modelo.')
    print('\t2) Consultar un modelo')
    print('\t3) Salir')
    print()


def process_menu_option(option, stub):
    if option == '1':
        name = input('Nombre del modelo >> ')
        proportion = input('Proporción del dataset a utilizar >> ')
        url = input('Dirección url del dataset (dejar en blanco para el dataset de leds) >> ')

        try:
            proportion = float(proportion)

            if url == "":
                url = URL

            print()
            create_model(stub, name, proportion, url)

        except ValueError:
            print('La entrada proporción debe ser un valor flotante de entre 0 y 1')

        except grpc.RpcError as e:
            print(f'{e.code().name}: {e.details()}')



    elif option == '2':
        name = input('Nombre del modelo >> ')
        leds = []
        print('0: Falso, 1: Verdadero')
        for i in range(7):
            led = input(f'Valor del led {i} >> ')
            try:
                led = bool(led)
                leds.append(led)
            except ValueError:
                print('La entrada de un led debe ser un valor de 1 o 0, para Verdadero o Falso respectivamente')

        try:
            print()
            consult_model(stub, name, leds)
        except grpc.RpcError as e:
            print(f'{e.code().name}: {e.details()}')


def run():
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = classifier_pb2_grpc.ClassifierStub(channel)

        try:
            while True:
                menu()
                opcion = input('Digite la opción deseada >>> ')
                if opcion == '3':
                    print('Saliendo...')
                    break
                process_menu_option(opcion, stub)
                print()

        except KeyboardInterrupt:
            print()
            print('Saliendo...')
            sys.exit(0)


def run_client():
    logging.basicConfig()
    run()

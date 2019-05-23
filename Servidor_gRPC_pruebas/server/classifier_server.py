
from concurrent import futures
import time
import logging

import grpc

import server.classifier_pb2
import server.classifier_pb2_grpc

_ONE_DAY_IN_SECONDS = 24 * 60 * 60

class ClassifierServicer(server.classifier_pb2_grpc.ClassifierServicer):

    def __init__(self):
        pass

    def CreateModel(self, request, context):
        pass

    def ConsultModel(self, request, context):
        pass


def serve():
    classifier_server = grpc.server(futures.ThreadPoolExecutor(max_workes=10))
    server.classifier_pb2_grpc.add_ClassifierServicer_to_server(
        ClassifierServicer(), classifier_server)

    classifier_server.add_insecure_port('[::]:50051')
    classifier_server.start()
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        classifier_server.stop(0)

if __name__ == '__main__':
    logging.basicConfig()
    serve()
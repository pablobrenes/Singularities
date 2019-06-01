# Proyecto programado para pasantía en Singularities

## Librerias necesarias:

El proyecto fue creado con las siguientes librerías:
- grpcio 1.16.1, para la creación del servicio gRPC.
- pandas 0.24.2, para obtener los valores del dataset en formato 'csv'.
- requests 2.21.0, para descargar los datasets desde el url provisto.
- pytorch 1.0.1, para la creación, entrenamiendo y consulta del modelo.

## Clasificador

El clasificador fue implementado con la librería [Pytorch](https://pytorch.org/). El código puede ser consultado en la
carpeta `classifier_model` donde existen dos archivos, `classifier.py` que contiene la Clase que representa el
clasificador y `data_loader.py` que contiene una Clase utilizada para cargar, leer y transformar los datos para el
entrenamiento.

### Arquitectura

El clasificador implementado es una red neuronal. Este posee una capa de entrada de 7 neuronas, una capa oculta de 21
neuronas y una capa de salida de 10 neuronas. Este modelo es implementado de manera sencilla en Pytorch con las
siguientes lineas de código.
```
model = nn.Sequential(
        nn.Linear(inputLayerSize, hiddenLayerSize),
        nn.Tanh(),
        nn.Linear(hiddenLayerSize, outputLayerSize),
        nn.Softmax(1))
```
A la salida de la capa oculta se le aplica la función _tangente hiperbólica_, y finalmente a la salida se le aplica la
función _softmax_ para clasificar la salida en 10 clases distintas.

### Entrenamiento

El modelo se entrena durante 4000 iteraciones, donde se optimizan los parámetros del modelo en cada iteración mediante
el método Adagrad, con un porcentaje de aprendizaje del 0.005%. La función utilizada para medir el error del modelo es
la entropia cruzada, pues es recomendada para la clasificación de N clases.

De igual manera el entrenamiento del modelo se implementa de manera sencilla en Pytorch, tal como se ve en el siguiente
codigo:
```
# Función de error a utilizar
error_function = nn.CrossEntropyLoss()

# Definir el método de optimización y la velocidad de aprendizaje
learning_rate = 0.005
optimizer = optim.Adagrad(model.parameters(), learning_rate)

# Ciclos necesarios para entrenar
epochs = 4000

for epoch in range(epochs):
    # Obtener la salida del modelo
    out = model(values)
    # Obtener el error del modelo
    error = error_function(out, targets)
    # Reiniciar el cálculo del gradiente para que no tome los cálculos realizados en interaciones previas
    optimizer.zero_grad()
    # Propagación del error hacia atrás para obtener el gradiente
    error.backward()
    # Optimizar los parámetros
    optimizer.step()
```

### Resultados del modelo

El modelo fue entrenado y probado 50 veces, para el entrenamiento se utilizó un 70% del dataset y para probarlo se
utilizó el restante 30%. Obteniendo los siguientes valores medios y sus desviaciones estándar.
- Tiempo de entrenamiento: 8.24 segundos de media con una desviación de 0.64 segundos.
- Porcentaje de muestras clasificadas correctamente: 73.10% de media, con una desviación de 0.15%.
- Error obtenido por la entropía cruzada: 1.73 de media, con desviación de 0.01.

Ahora, aunque el modelo siguiese siendo entrenado por encima de las 4000 iteraciones, no se percibió un aumento
importante en el porcentaje de muestras correctamente clasificadas. Aunque este fue de esperarse antes de crear el
modelo.

Antes de la creación del modelo, cuando se analizó el dataset, se encontró que para la misma combinación de leds,
existían salidas distintas. Por ejemplo, para la combinación (1, 1, 1, 1, 1, 1, 1) se encontraron 7 valores distintos:
487 entradas marcadas como 8, 54 marcadas como 6, 60 como 0, 52 como 9, 6 como 2, 4 como 3 y finalmente 7 como 5. Este
fenómeno también se presentó con otras combinaciones de leds distintas.

Se puede concluir que el dataset presenta bastante ruido. Por lo que, el modelo aun con un valor porcentual de muestras
clasificadas correctamente de 73%, es un modelo aceptable ya que, fue entrenado sin eliminar el ruido. Y al utilizar las
mismas muestras con ruido para probarlo es esperable este resultado. 

## Servicio gRPC

El archivo en formato .proto, puede verse completo en el archivo `classifier.proto`. En resumen, se crea un servicio con
dos llamadas, una para crear un modelo y otra para consultar el modelo. La primer llamada se comunica recibiendo un
_message_ de tipo _CreateModelRequest_ y retornando un _message_ de tipo _ModelRepresentation_. Similarmente la segunda
llamada recibe un _message_ de tipo _ConsultModelRequest_ y retorna un _message_ de tipo _ConsultModelResponse_.

Donde cada mensaje tiene los siguientes campos:
- CreateModelRequest: un nombre (string), una proporción (float) y un url (string).
- ModelRepresentation: un detalle (string).
- ConsultModelRequest: un nombre (string) y una lista de valores booleanos (repeated bool)
- ConsultModelResponse: una salida (int32).

Una vez definido este archivo, se procede a utilizar el compilador de proto para Python. Obteniendo dos archivos:
`classifier_pb2.py` y `classifier_pb2_grpc.py`, el primero contiene las descripciones de los mensajes y el servicio
necesario para implementar el servicio gRPC en Python, mientras que el segundo contiene dos interfaces para implementar
dos clases, una para implementar el servidor y la segunda para invocar los métodos remotos.

### Servidor

El servidor se implementa en una clase según la interfaz provista por el compilador de proto. Se reescriben los métodos
definidos según las llamadas en el servicio gRPC.

#### Llamada para crear un modelo
El código puede ser encontrado en el archivo `classifier_server`, en el método _CreateModel_ de la clase
_ClassifierServicer_

El servidor, cuando recibe una petición de crear un modelo, se encarga de crear un nombre de archivo basado en el nombre
del modelo donde se guardará el modelo en disco para posteriores consultas, seguidamente crea una representación de la
entidad asociada al modelo para almacenarla en la base de datos, luego, agrega a la cola de trabajo la representación
de la entidad para que algún trabjador la tome y comience el entrenamiento del modelo. Finalmente envía al cliente una
representación del modelo y termina la llamada.

El proceso recién descrito implica que no hubo errores. En caso de que el servidor detecte un error, como qué los campos
del mensaje no tienen los valores correctos o bien el nombre del modelo ya está siendo utilizado el servidor retorna
un _mesagge_ de tipo _ModelRepresentation_ vació, pero actualiza la variable de contexto de la llamada del servicio
gRPC para indicar  que hubo un error, estas validaciones pueden ser consultadas en el archivo `classifier_server`, en el
 método _validate_create_model_request_ de la clase _ClassifierServicer_.

#### Llamada para consultar un modelo
El código puede ser encontrado en el archivo `classifier_server`, en el método _ConsultModel_ de la clase
_ClassifierServicer_

Al momento de recibir esta llamada el servidor primero busca la entidad asociada al modelo en la base de datos indicado
según el nombre, obtiene la localización del archivo que representa los parámetros del modelo, crea un nuevo
clasificador y le carga los parámetros según la localización del archivo, le realiza una consulta al modelo y regresa al
cliente la salida obtenida del modelo.

Similarmente en la llamada anterior, el proceso anterior es el que no implica fallos, si los valores de leds son
incorrectos, o bien se quiere consultar un modelo que no exista, o que no esté entrenado aún, se retorna un
_mesagge_ de tipo _ConsultModelResponse_ vació y se actualiza el contexto. Estas validaciones pueden ser consultadas en
el archivo `classifier_server`, en el método _validate_consult_model_request_ de la clase _ClassifierServicer_.

#### Servicio
Cuando el servidor inicia se encarga de verificar que todos los recursos del servidor estén en orden, y si aplica,
procede a cargar los necesarios.

Luego de esto el servidor crea la cola de trabajo y lanza los trabajadores asociados a esta para que se encarguen
de entrenar los modelos.

Crea los servicios necesarios según la librería grpc, agrega un puerto (actualmente 50051) y comienza a esperar
peticiones.

El servicio puede ser detenido pulsando ctrl + c desde la consola donde se lanzó el servicio. Cuando esta combinación
es recibida por el servidor, el servidor espera a que la cola sea procesada, o bien que los trabajadores terminen. Y
finalmente detiene el servidor. El servidor actualmente no contempla que se detenga abruptamente, es necesario esperar
a que la cola sea totalmente procesada, el servidor se terminará por si mismo una vez esto suceda.

#### Recursos del servidor

**Estructura de archivos**: 
El servidor requiere de una carpeta principal, _serverdata_, dentro de esta deben existir dos más,
_temp_datasets_storage_ y _models_, donde se almacenan los datasets temporalmente cuando se descargan y los modelos
entrenados, respectivamente. Además dentro de la carpeta principal debe existir un archivo de formato _json_ bajo el
nombre de _model_entity_database.json_.

El servidor se encarga de crear estas carpetas y archivos al momento de ejecutarlo si estas no existen.

**Base de datos**: 
Esta es lo más simple posible, al momento de ejecución es una lista de Python, mientras que en disco, se
representa como un archivo de formato _json_. Que se carga una única vez al momento de levantar el servidor. Esta se
actualiza en disco cada vez que un modelo fue entrenado.

Existen tres funciones asociadas a esta base de datos, una para obtener una entidad, agregar una entidad y actualizar
una entidad para indicar que ya fue entrenada específicamente. Pueden verse en el archivo `classifier_server`, son las
tres primeras funciones definidas: _get_model_entity_, _add_model_entity_, y _update_trained_value_.

**Trabajadores y cola de trabajo**:
Cuando el servidor inicia, lanza una cantidad definida de hilos con el propósito de entrenar modelos. Estos están
asociados a una cola de trabajo, implementada con _Queue_ del paquete _queue_ de Python, que ofrece colas
especializadas para ser utilizadas por múltiples hilos de ejecución.

La cola de trabajo recibe entidades asociadas al modelo. Definidas por la clase _ModelEntity_ (puede verse en 
`classifier_resources.py`), Estas contienen el nombre del modelo, la proporción del dataset a usar, el url origen,
la localización del archivo en memoria donde se almacenan los parámetros del modelo, fecha de creación y finalmente
un valor booleano que indica si el modelo en particular ya fue entrenado o no, para que el servidor sepa si ya puede o 
no consultar un modelo de ser necesario.

El trabajo que realiza un trabajador es el siguiente: obtener la entidad de la cola de trabajo, descargar el archivo
asociado al dataset, crear un clasificador no entrenado, obtener la proporción necesaria del dataset y comenzar el
entrenamiento del modelo, una vez entrenado el modelo elimina el archivo del dataset, finalmente guarda el modelo en
disco, actualiza el valor de entrenado en la base de datos y le indica a la cola que ha completado el trabajo. El 
código del trabajador puede ser consultado en el archivo `classifier_server`, en la función _train_model_worker_.

### Cliente
El cliente es un pequeño programa de consola, con un menú simple para utilizar las dos llamadas especificadas en el 
servicio gRPC.

El cliente implementa una clase obtenida por el compilador de proto para Python. Basta instanciar
esta clase utilizando como parámetro el canal creado con la librería grpc con la dirección y el puerto del servidor.
Una vez instanciada esta clase todas las llamadas especificadas en el servicio gRPC pueden ser invocadas a través de
esta instancia, utilizando los objetos definidos también por el compilador de proto, como se mencionó en la sección 
Servicio gRPC.

Para el manejo de errores en el cliente, para Python, basta encerrar la llamada en un bloque _try-except_. Si el
servidor actualiza el contexto a un error, automaticamente la llamada dentro del cliente lanzará una excepción que puede
ser capturada con `except grpc.RpcError`, para posteriormente manejarlo.
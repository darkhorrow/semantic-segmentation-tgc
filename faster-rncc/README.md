# Detección de dorsales con Faster RCNN

En este directorio se encuentra una adaptación del Faster RCNN originalmente de Pablo Hernández, el cuál desarrolló como su trabajo final de grado. Esta adaptación distribuye los elementos de manera que sea más legible, además de solucionar errores cuando se proporcionaban imágenes pequeñas.

Junto con el Faster RCNN , se incluyen

A continuación, se explica la funcionalidad de cada fichero e información miscelánea de interés, partiendo de la siguiente estructura:

    faster-rcnn
    ├── rbnd
    │   ├── rbnd_model
    │   ├── utils
    │   └── run.py
    ├── results
    ├── utils
    ├── README.md
    ├── RunFasterRCNN.ipynb
    └── TrainFasterRCNN.ipynb

<ins>**rbnd**</ins>

Este directorio consiste en el proytecto Python que contiene todo el código necesario apra realizar la detección (suponiendo que se tiene el modelo ya entrenado y el fichero), siendo *run.py* el punto de entrada de la aplicación.

<ins>**results**</ins>

En este directorio se contiene ficheros obtenidos de los procesos de los *Jupyter notebook*, principalemte de los siguientes tipos:

* Predicciones obtenidas por Faster RCNN
* Métricas de las predicciones obtenidas por imagen tratada
* Métricas globales

<ins>**utils**</ins>

Este directorio contiene ficheros de utilidad varios que permiten realizar alguna tarea de preprocesamiento para **rbnd**.

<ins>**RunFasterRCNN.ipynb**</ins>

Este *Jupyter notebook* contiene la llamada al *run.py* de *rbnd*. Se utilzó inicialmente para hacer pruebas ejecutándolo con las GPU proporcionadas por *Google Colab*

<ins>**TrainFasterRCNN.ipynb**</ins>

Este *Jupyter notebook* contiene el código de entrenamiento de Faster R-CNN.

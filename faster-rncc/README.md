# Detección de dorsales con Faster RCNN

En este directorio se encuentra una adaptación del Faster RCNN originalmente de Pablo Hernández, el cuál desarrolló como su trabajo final de grado. Esta adaptación distribuye los elementos de manera que sea más legible, además de solucionar errores cuando se proporcionaban imágenes pequeñas.

Junto con el Faster RCNN , se incluyen

A continuación, se explica la funcionalidad de cada fichero e información miscelánea de interés, partiendo de la siguiente estructura:

    faster-rcnn
    ├── output_result_files
    │   └── ...
    ├── rbnd
    │   ├── rbnd_model
    │   ├── utils
    │   └── run.py
    ├── MethodComparison.ipynb
    ├── README.md
    └── RunFasterRCNN.ipynb

<ins>**output_result_files**</ins>

En este directorio se contiene ficheros obtenidos de los procesos de los *Jupyter notebook*, principalemte de los siguientes tipos:

* Predicciones obtenidas por Faster RCNN
* Métricas de las predicciones obtenidas por imagen tratada
* Métricas globales

<ins>**rbnd**</ins>

Este directorio consiste en el proytecto Python que contiene todo el código necesario apra realizar la detección (suponiendo que se tiene el modelo ya entrenado y el fichero), siendo *run.py* el punto de entrada de la aplicación.

<ins>**MethodComparison.ipynb**</ins>

Este *Jupyter notebook* se encarga de graficar los resultados presente en *output_result_files* mediante la librería *Matplotlib*, principalmente.

<ins>**RunFasterRCNN.ipynb**</ins>

Este *Jupyter notebook* contiene la llamada al *run.py* de *rbnd*. Se utilzó inicialmente para hacer pruebas ejecutándolo con las GPU proporcionadas por *Google Colab*

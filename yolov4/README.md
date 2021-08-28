# Extracción de personas con Yolov4

En este directorio se encuentra el contenido relativo a la extracción de personas en imágenes con el fin de usar estas imágenes en la parte de segmentación semántica.

A continuación, se explica la funcionalidad de cada fichero e información miscelánea de interés, partiendo de la siguiente estructura:

    yolov4
    ├── results
    │   ├── custom
    │   └── tgcrbnw 
    ├── Crop_YOLOv4_People.ipynb
    ├── README.md
    └── YOLOv4_People_Detection.ipynb

<ins>**results**</ins>

En este directorio se contiene ficheros obtenidos de los procesos de los *Jupyter notebook* para los dos conjuntos de datos usados:

* **images.txt**: Es el listado de imágenes. Lo usa Yolov4 para realizar múltiples predicciones en una misma ejecución.

* **result_90.json**: Resultados originales de Yolov4.

* **transform_90.csv**: Resultado de transformar *result_90.json* al formato usado en el *Faster RCNN* mediante *yolo2faster_rcnn_format.py*, presente en otro directorio (bib-numbers).

<ins>**Crop_YOLOv4_People.ipynb**</ins>

Este es el *Jupyter notebook* en el cuál se usan las coordenadas de los *bounding boxes* de *transform_90.csv* para recortar las imágenes y obtener únicamente a las personas detectadas.

<ins>**YOLOv4_People_Detection.ipynb**</ins>

Este *Jupyter notebook* se encarga de usar Yolov4 para la generación del fichero *result_90.json*, usando los pesos preentrenados de COCO. Se usa un umbral de confianza del 90% con el fin de descartar a personas que se encuentran en el fondo de la foto que no son relevantes en lo que a las maratones respecta.

Para ello, se debe tener el fichero *images.txt* previamnete, o uno que cumpla el formato.

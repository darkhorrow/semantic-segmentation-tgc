# Segmentación semántica con Deeplabv3

En este directorio se encuentra el contenido relativo a la segmentación semántica realizada sobre las imágenes de las maratones, con el fin de obtener un conjunto de datos con imágenes de menor dimensión, que facilitaran y mejorarán los resultados de la detección de dorsales.

A continuación, se explica la funcionalidad de cada fichero e información miscelánea de interés, partiendo de la siguiente estructura:

    deeplabv3
    ├── deeplab_extras
    │   ├── data_generator.py
    │   └── requirements.txt
    ├── Deeplab+LIP.ipynb
    ├── README.md
    ├── UseDeeplabCustomModel.ipynb
    └── UseDeeplabModel.ipynb

<ins>**deeplab_extras**</ins>

En este directorio se contiene ficheros a sustituir dentro del repositorio de Deeplabv3 o para su instalación local. Actualmente, dichos ficheros son:

* **data_generator.py**: Este fichero debe sustituirse por el original que se clona del repositorio de Deeplabv3 en *Deeplab+LIP.ipynb*. En él se define la informaciónb del conjunto de datos LIP y CIHP (no usado en la versión final del proyecto).

* **requirements.txt**: Este fichero contiene todas las dependencias instaladas en el entorno de *Google Colab* con el que se desarrolló el proyecto. Puede usarse para hacer una instalación local.

<ins>**Deeplab+LIP.ipynb**</ins>

Este es el *Jupyter notebook* en el cuál se realiza el entrenamiento de la red, partiendo de los pesos preentrenados de Mobilenetv2, así como la exportación del modelo final, el cuál se usa en *UseDeeplabCustomModel.ipynb*.

Además, se realiza la extracción y conversión del conjunto de datos LIP a tfrecord.

<ins>**UseDeeplabCustomModel.ipynb**</ins>

Este *Jupyter notebook* se encarga de usar el modelo entrenado en este proyecto y realizar recortes de las imágenes, de manera que se trata de obtener únicamente la parte correspondiente al torso y pantalones.

Si la segmentación no se puede realizar o la zona en la que se detecta el torso/pantalones no coincide con el caso esperado de un corredor, se obtiene la imagen intacta.

<ins>**UseDeeplabModel.ipynb**</ins>

Este *Jupyter notebook* es el demo original de Deeplabv3 con algunas adaptaciones para usar un conjunto de datos de corredores y los de LIP con los modelos preentrenados que ofrece dicho demo.

## Pasos para la instalación y ejecución de Deeplabv3 localmente

Si bien estos pasos no se realizaron durante el proyecto, ya que se hizo uso de *Google Colab* junto a los *Jupyter notebook*, indican lo que se realiza en ellos y cómo se esperaría que funcionen en un entorno de Python 3 estándar.

### Instalación de las dependencias

En el fichero ***deeplab_extras/requirements.txt*** se encuentra un *pip freeze* tomado del entorno de Deeplab en el que funciona correctamente. Para instalar las dependencias, se usa el siguiente comando:

    pip install -r requirements.txt

### Descarga y configuración de Deeplabv3

A continuación, se puede clonar el repositorio que contiene Deeplabv3 mediante el siquiente comando:

    git clone https://github.com/tensorflow/models

Una vez se disponga del repositorio, se debe cambiar el PYTHONPATH de la siguiente manera:

    %env PYTHONPATH=/path/absoluto/models/research/:/path/absoluto/models/research/deeplab:/path/absoluto/models/research/slim:/env/python

---
**Nota:** ¡Este paso es importante! Si no se realiza, Deeplabv3 no funcionará posteriormente.

---

A continuación, tenemos que definir nuestro dataset en el fichero ***deeplab_extras/dataset_generator.py***. Por simplicidad, se adjunta con este documento. Se encuentra en models/research/deeplab/datasets.

### Obtención y transformación del dataset (LIP)

El dataset de LIP se encuentra en el siguiente Google Drive: https://drive.google.com/drive/folders/1ZjNrTb7T_SsOdck76qDcd5OHkFEU0C6Q

De todos los ficheros *.zip* que hay, sólo interesan dos de ellos: **TrainVal_images.zip** y **TrainVal_parsing_annotations.zip**. A su vez, dentro de estos ficheros comprimidos hay otros *.zip*: también se descomprimen.

Una vez se posean todos los ficheros descomprimidos, creamos dos carpetas, las cuáles tendrán los tfrecord de los dataset de *train* y *test*.

Por ejemplo:

    mkdir train_lip_tfrecord
    mkdir val_lip_tfrecord

También se puede aprovechar las carpetas *trainval_lip_tfrecord*, *eval_results_lip* y *checkpoint*, que se usan más adelante para el entrenamiento y visualización.

#### Creación de los tfrecord

Para el conjunto de entenamiento:

    python models/research/deeplab/datasets/build_voc2012_data.py \
    --image_folder="/path/absoluto/imágenes/TrainVal_images/train_images" \
    --semantic_segmentation_folder="/path/absoluto/TrainVal_parsing_annotations/train_segmentations" \
    --list_folder="/path/absoluto/imágenes" \
    --image_format="jpg" \
    --output_dir="train_lip_tfrecord/"

Para el conjunto de test:

    python models/research/deeplab/datasets/build_voc2012_data.py \
      --image_folder="/path/absoluto/imágenes/TrainVal_images/val_images" \
      --semantic_segmentation_folder="/path/absoluto/TrainVal_parsing_annotations/val_segmentations" \
      --list_folder="/path/absoluto/imágenes" \
      --image_format="jpg" \
      --output_dir="val_lip_tfrecord/"

Una vez se tengan ambas carpetas con los *tfrecord*, se procede a crear una tercera carpeta que contenga los ficheros de train y test. También hay que cambiar el nombre de todos los tfrecord de manera que el prefijo *train_id/val_id* pase a ser *train/val*.


### Verificación de la instalación

Antes de realizar, el entrenamiento como tal, se puede verificar que el entorno de Deeplabv3 es correcto ejecutando los tests:

    cd /models
    python deeplab/model_test.py

Si ejecuta los tests, se sabe que el entorno es correcto.

### Entrenamiento

Si no existe ningún problema, el siguiente comando comienza el entrenamiento con el dataset LIP:

    !python deeplab/train.py --logtostderr --training_number_of_steps=10000 --train_split="train" --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size="513,513" --train_batch_size=1 --dataset="lip" --train_logdir="/path/absoluto/checkpoint" --dataset_dir="/path/absoluto/trainval_lip_tfrecord/" --fine_tune_batch_norm=false --initialize_last_layer=true --last_layers_contain_logits_only=false

### Visualización

De manera análoga al entrenamiento, se puede realizar la visualización del conjunto de test de la siguiente manera:

    !python deeplab/vis.py --logtostderr --vis_split="val" --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --vis_crop_size="513,513" --dataset="lip" --checkpoint_dir="/path/absoluto/checkpoint" --vis_logdir="/path/absoluto/eval_results_lip" --dataset_dir="/path/absoluto/trainval_lip_tfrecord" --max_number_of_iterations=1 --eval_interval_secs=0

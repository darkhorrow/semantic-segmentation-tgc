import glob
import os.path
import pickle
import time

import cv2
import numpy as np
import pandas as pd
# Estaba contensorflow 1.14.0
# actualizo a 2.4.1, luego 2.3 porque está cuda 10.1 en servidor
# fijo LD_LIBRARY_PATH para cuda 10.1 y cudnn 7.6.4
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from .utils.rbndd_utils import *
from .rbnd_model.base_model import nn_base
from .rbnd_model.rpn_model import rpn_layer
from .rbnd_model.classifier_model import classifier_layer


tf.config.list_physical_devices('GPU')

# Detector
# Adaptado a portatil
base_path = '../Modelos/TFG/ModeloTGCdia/'
config_output_filename = os.path.join(base_path, 'model/model_vgg_config_mode.pickle')  # Original de Pablo sin mode

# Original Pablo
# base_path = './'
# config_output_filename = os.path.join(base_path, 'model/model_vgg_config.pickle') # Original de Pablo sin mode

with open(config_output_filename, 'rb') as f_in:
    C = pickle.load(f_in)

# capa Input del modelo VGG (Imagenes RGB)
img_input = Input(shape=(None, None, 3))
# capa Input del modelo RoI Pooling
roi_input = Input(shape=(C.num_rois, 4))
# capa Input del modelo clasificador (convolutional feature map (H/stride, W/stride, 512))
num_features = 512
feature_map_input = Input(shape=(None, None, num_features))

# define la red base (VGG16)
shared_layers = nn_base(img_input)

# define el modelo RPN
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn = rpn_layer(shared_layers, num_anchors)

# define el modelo clasificador final
classifier = classifier_layer(feature_map_input, roi_input, C.num_rois, nb_classes=len(C.class_mapping))

# Creamos los modelos
model_rpn = Model(img_input, rpn)
model_classifier = Model([feature_map_input, roi_input], classifier)

print('Loading weights from {}'.format(C.model_path))
# model_rpn.load_weights(C.model_path, by_name=True)
model_rpn.load_weights("./model/model_frcnn_vgg.hdf5", by_name=True)
# model_classifier.load_weights(C.model_path, by_name=True)
model_classifier.load_weights("./model/model_frcnn_vgg.hdf5", by_name=True)

# se intercambian las parejas <'clase', valor>
class_mapping = C.class_mapping
class_mapping = {v: k for k, v in class_mapping.items()}
print(class_mapping)

test_base_path = 'C:/Users/otsed/Desktop/Databases/TGC'  # Directory from load the test images
# imgs_path = os.listdir(test_base_path)
all_imgs = []
classes = {}

# threshold score (se ignoran las predicciones con valores de probabilidad menores)
bbox_threshold = 0.7

# las predicciones se alamcenan en un dataframe
column_names = ["name", "xmin", "ymin", "xmax", "ymax", "class", "score"]
predictions = pd.DataFrame(columns=column_names)

# lee annotationTest.txt en un Dataframe para marcar los ground-truth bboxes en las imagenes
if os.path.exists(test_base_path + '/annotateTest.txt'):
    df_bboxes_gt_test = pd.read_csv(test_base_path + '/annotateTest.txt', sep=",", header=None)
    df_bboxes_gt_test.columns = ["filename", "xmin", "ymin", "xmax", "ymax", "class"]
    # agrupa las anotaciones de una misma imagen (columna "filename")
    gt_grouped_by_filename = df_bboxes_gt_test.groupby('filename')

debug = 0  # 1 to show detections

################################################
# Images location
DatasetPath = "C:/Users/otsed/Desktop/Databases/TGC/"
DatasetPath = "E:/Datasets/TransGC2020/TGCC-2020-Frames/AnotadaArucas/Frames/"
DatasetPath = "E:/Datasets/TransGC2020/TGCC-2020-Frames/AnotadaTeror/Frames/"
# DatasetPath = "E:/Datasets/TransGC2020/TGCC-2020-Frames/AnotadaPresadeHornos/Frames/"
# DatasetPath = "E:/Datasets/TransGC2020/TGCC-2020-Frames/AnotadaAyagaures/Frames/"
# DatasetPath = "E:/Datasets/TransGC2020/TGCC-2020-Frames/AnotadaParqueSur/Frames/"

# CVSPORTS
DatasetPath = "F:/Datasets/TGCRBNWv0.1UnifiedImgs"
results_base_path = "F:/Datasets/FasterRCNNResults"
nframes = 0

for imgfile in glob.glob(DatasetPath + "/*.jpg"):
    # Load images names
    nframes += 1
    # results
    bibfile = imgfile.replace(".jpg", "_bibs.txt")
    predictions = []

    if os.path.exists(bibfile) == False:

        img = cv2.imread(imgfile);
        # detfile = imgfile.replace(".jpg", "_RBNdeetction.txt")
        # fid = open(detfile, "w")
        # Captura fotograma
        t = time.time()

        # re-escala la imagen y transforma BGR -> RGB
        X, ratio = format_img(img, C)

        # Y1: probabilidad de cada anchor (de incluir un objeto) correspondiente a cada punto del feature map
        # Y2: deltas del bbox de cada anchor correspondiente a cada punto del feature map
        # Los valores deltas son codificados con la varianza, p.e. x=(x_gt-x_anc)/(w_anc*var) y w=ln(w_gt/w_anc)/var
        # F: feature map
        [Y1, Y2, F] = model_rpn.predict(X)

        # Corrige los anchores con las predicciones delta del modelo RPN y selecciona bboxes mediante NMS
        # R.shape = (300, 4)
        R = rpn_to_roi(Y1, Y2, C, overlap_thresh=0.7)

        # (x1,y1,x2,y2) => (x,y,w,h)
        R[:, 2] -= R[:, 0]
        R[:, 3] -= R[:, 1]

        # almacena info de los ROI seleccionados
        bboxes = {}
        probs = {}

        for jk in range(R.shape[0] // C.num_rois + 1):
            # selecciona los siguientes 4 bboxes
            ROIs = np.expand_dims(R[C.num_rois * jk:C.num_rois * (jk + 1), :], axis=0)
            if ROIs.shape[1] == 0:
                break

            if jk == R.shape[0] // C.num_rois:
                # pad R para incluir 4 ROIs que es la entrada esperada por el clasificador final
                curr_shape = ROIs.shape
                target_shape = (curr_shape[0], C.num_rois, curr_shape[2])
                ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
                ROIs_padded[:, :curr_shape[1], :] = ROIs
                ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
                ROIs = ROIs_padded

            # F: feature maps
            # P_cls (4x2): score de cada ROI (4 ROI de entrada) y para cada clase (incluyendo la 'bg')
            # P_regr (4x4): deltas bbox (4 values) para cada clase y para cada ROI (4 ROI de entrada)
            [P_cls, P_regr] = model_classifier.predict([F, ROIs])

            # Calcula coordenadas bboxes en la imagen original
            for ii in range(P_cls.shape[1]):
                # Ignora ROI con (score<bbox_threshold) or (ROI con clase 'bg')
                cls_num = np.argmax(P_cls[0, ii, :])
                if np.max(P_cls[0, ii, :]) < bbox_threshold or cls_num == (P_cls.shape[2] - 1):
                    continue

                cls_name = class_mapping[cls_num]  # nombre asignado a la clase
                if cls_name not in bboxes:
                    bboxes[cls_name] = []
                    probs[cls_name] = []

                (x, y, w, h) = ROIs[0, ii, :]
                try:
                    # extrae deltas predecidos por el clasificador final para este ROI
                    (tx, ty, tw, th) = P_regr[0, ii, 4 * cls_num:4 * (cls_num + 1)]
                    tx /= C.classifier_regr_std[0]
                    ty /= C.classifier_regr_std[1]
                    tw /= C.classifier_regr_std[2]
                    th /= C.classifier_regr_std[3]
                    # corregimos bbox del ROI
                    x, y, w, h = apply_regr_classfinal(x, y, w, h, tx, ty, tw, th)
                except:
                    pass

                # almacenamos coordenadas de bboxes y scores del ROI
                bboxes[cls_name].append(
                    [C.rpn_stride * x, C.rpn_stride * y, C.rpn_stride * (x + w), C.rpn_stride * (y + h)])
                probs[cls_name].append(np.max(P_cls[0, ii, :]))

            # aplica NMS sobre los bboxes detectados y dibuja el resultado en la imagen
        all_dets = []  # almacena las detecciones
        fid = open(bibfile, "w")

        for key in bboxes:
            bbox = np.array(bboxes[key])

            new_boxes, new_probs = non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.2)

            for jk in range(new_boxes.shape[0]):
                (x1, y1, x2, y2) = new_boxes[jk, :]

                # Calcula coordenadas en la imagen original y dibuja bbox detectado
                (real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)
                cv2.rectangle(img, (real_x1, real_y1), (real_x2, real_y2), (0, 0, 255), 2)
                # detection data to txt file coordinates and confidence
                fid.write("%d %d %d %d %d\n" % (real_x1, real_y1, real_x2, real_y2, int(100 * new_probs[jk])))

                textLabel = '{}: {}'.format("confianza", int(100 * new_probs[jk]))
                all_dets.append((key, 100 * new_probs[jk]))

                # muestra el string "textLabel" junto al bbox detectado
                (retval, baseLine) = cv2.getTextSize(textLabel, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)
                textOrg = (real_x1, real_y1)
                xxx1 = textOrg[0] - 0
                yyy1 = textOrg[1] + baseLine - 0
                xxx2 = textOrg[0] + retval[0] + 0
                yyy2 = textOrg[1] - retval[1] - 0
                if xxx1 < 0 or yyy1 < 0 or xxx2 < 0 or yyy2 < 0:
                    textOrg = (real_x2, real_y2)
                    xxx1 = textOrg[0] - retval[0] - 0
                    yyy1 = textOrg[1] + retval[1] + 0
                    xxx2 = textOrg[0] + 0
                    yyy2 = textOrg[1] - baseLine + 0
                    textOrg = (xxx1, yyy1 - baseLine)

                cv2.rectangle(img, (xxx1, yyy1), (xxx2, yyy2), (0, 0, 0), 1)
                cv2.rectangle(img, (xxx1, yyy1), (xxx2, yyy2), (255, 255, 255), -1)
                cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1)

                predictions.append(
                    {"name": imgfile, "xmin": real_x1, "ymin": real_y1, "xmax": real_x2, "ymax": real_y2, "class": key,
                     "score": int(100 * new_probs[jk])})

            # almacena la imagen en disco
            aux_filename = DatasetPath
            print(results_base_path + '/' + (imgfile.split(aux_filename)[1])[1:])
            cv2.imwrite(results_base_path + '/' + (imgfile.split(aux_filename)[1])[1:], img)

            with open(results_base_path + '/' + ((imgfile.split(aux_filename)[1])[1:]).replace('.jpg', '.txt'),
                      'w') as p_file:
                for prediction in predictions:
                    p_file.write(
                        f'{prediction["name"]} {prediction["xmin"]} {prediction["ymin"]} {prediction["xmax"]} {prediction["ymax"]} {prediction["class"]} {prediction["score"]}\n')

        fid.close()

        if debug:
            print("tiempo de procesamiento : {:.3f}".format(time.time() - t))
            cv2.imshow("Normalized", img)

    # Finaliza pulsando q
    tec = cv2.waitKey(4)
    # ESc para terminar
    if tec & tec == 27:  # Esc
        break

# Libera al detener
cv2.destroyAllWindows()

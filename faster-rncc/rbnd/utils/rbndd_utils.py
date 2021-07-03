import numpy as np
import math


def non_max_suppression_fast(boxes, probs, overlap_thresh=0.9, max_boxes=300):
    """NMS algorithm to avoid duplicates in the bounding boxes delimiting the same object
    Code extracted from: http://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/

    Step 1: Order probabilities list
    Step 2: Get the highest probability and move it to a new list
    Step 3: Calcualte IoU between the bounding box with the previous probability and the other from the list
    Step 4: Repeat 2 and 3 until the probabilities list is empty
    """

    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    np.testing.assert_array_less(x1, x2)
    np.testing.assert_array_less(y1, y2)

    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    pick = []

    area = (x2 - x1) * (y2 - y1)

    idxs = np.argsort(probs)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1_int = np.maximum(x1[i], x1[idxs[:last]])
        yy1_int = np.maximum(y1[i], y1[idxs[:last]])
        xx2_int = np.minimum(x2[i], x2[idxs[:last]])
        yy2_int = np.minimum(y2[i], y2[idxs[:last]])

        ww_int = np.maximum(0, xx2_int - xx1_int)
        hh_int = np.maximum(0, yy2_int - yy1_int)

        area_int = ww_int * hh_int
        area_union = area[i] + area[idxs[:last]] - area_int

        overlap = area_int / (area_union + 1e-6)

        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_thresh)[0])))

        if len(pick) >= max_boxes:
            break

    boxes = boxes[pick].astype("int")
    probabilities = probs[pick]
    return boxes, probabilities


# aplica la correccion de los deltas predichos por el modelo RPN
def apply_regr_rpn(X, T):
    # corrige las coordenadas (x,y,w,h) del anchor segun los deltas (tx,ty,tw,th)
    # Segun se indica en el paper original:
    # tx=(cx_gt-cx_anchor)/w_anchor, ty=(cy_gt-cy_anchor)/h_anchor, tw=log(w_gt/w_anchor), tw=log(h_gt/h_anchor)
    # Nota: np.exp() permite trabajar con arrays, mientras que math.exp() solo con escalares
    try:
        x = X[0, :, :]
        y = X[1, :, :]
        w = X[2, :, :]
        h = X[3, :, :]

        tx = T[0, :, :]
        ty = T[1, :, :]
        tw = T[2, :, :]
        th = T[3, :, :]

        cx = x + w / 2.
        cy = y + h / 2.
        cx1 = tx * w + cx
        cy1 = ty * h + cy

        w1 = np.exp(tw.astype(np.float64)) * w
        h1 = np.exp(th.astype(np.float64)) * h
        x1 = cx1 - w1 / 2.
        y1 = cy1 - h1 / 2.

        x1 = np.round(x1)
        y1 = np.round(y1)
        w1 = np.round(w1)
        h1 = np.round(h1)
        return np.stack([x1, y1, w1, h1])
    except Exception as e:
        print(e)
        return X


# aplica la correccion de los deltas predichos por el modelo clasificador final
def apply_regr_classfinal(x, y, w, h, tx, ty, tw, th):
    # corrige las coordenadas (x,y,w,h) del anchor segun los deltas (tx,ty,tw,th)
    # tx=(cx_gt-cx_anchor)/w_anchor, ty=(cy_gt-cy_anchor)/h_anchor, tw=log(w_gt/w_anchor), tw=log(h_gt/h_anchor)
    try:
        cx = x + w / 2.
        cy = y + h / 2.
        cx1 = tx * w + cx
        cy1 = ty * h + cy

        w1 = math.exp(tw) * w
        h1 = math.exp(th) * h
        x1 = cx1 - w1 / 2.
        y1 = cy1 - h1 / 2.

        x1 = int(round(x1))
        y1 = int(round(y1))
        w1 = int(round(w1))
        h1 = int(round(h1))

        return x1, y1, w1, h1

    except ValueError:
        return x, y, w, h
    except OverflowError:
        return x, y, w, h
    except Exception as e:
        print(e)
        return x, y, w, h


# define los ROIs a partir de las predicciones de scores y deltas de cada anchor por el modelo RPN
def rpn_to_roi(out_rpn_cls, out_rpn_regr, C, max_boxes=300, overlap_thresh=0.9):
    # Pasos:
    #   1. Calcula los bboxes de los ROIs: obtiene coordenadas de los anchores de cada punto del feature map
    #   2. Cada anchor es corregido por los deltas predichos por el modelo RPN
    #   3. Recorta aquellos bboxes que sobresalgan de la imagen
    #   4. Aplica NMS sobre los bboxes
    # Devuelve las coordenadas de los bboxes seleccionados (no los scores)

    # Decodificacion deltas (deltas = deltas*0.25) - p.e. x=(x_gt-x_anc)/(w_anc*var) y w=ln(w_gt/w_anc)/var
    out_rpn_regr = out_rpn_regr / C.std_scaling

    anchor_sizes = C.anchor_box_scales  # (son 3)
    anchor_ratios = C.anchor_box_ratios  # (son 3)

    assert out_rpn_cls.shape[0] == 1
    (rows, cols) = out_rpn_cls.shape[1:3]

    # A.shape = (4, feature_map.height, feature_map.width, num_anchors) = (4,18,25,9) si la imagen es 400x300
    # A almacena las coordenadas de los 9 anchores por cada punto del feature map => 18x25x9=4050 anchores
    A = np.zeros((4, out_rpn_cls.shape[1], out_rpn_cls.shape[2], out_rpn_cls.shape[3]))

    curr_anchor = 0  # indica un anchor en el rango 0~8 (9 anchores)
    for anchor_size in anchor_sizes:
        for anchor_ratio in anchor_ratios:
            # ancho y alto del anchor en el feature map = (ancho * escala) / 16
            anchor_x = (anchor_size * anchor_ratio[0]) / C.rpn_stride
            anchor_y = (anchor_size * anchor_ratio[1]) / C.rpn_stride

            # regr almacena los deltas del current_anchor en todas las posiciones del feature map
            regr = out_rpn_regr[0, :, :, 4 * curr_anchor:4 * curr_anchor + 4]  # shape => (18, 25, 4)
            regr = np.transpose(regr, (2, 0, 1))  # shape => (4, 18, 25)

            # Grid del mismo tamano que el feature map
            X, Y = np.meshgrid(np.arange(cols), np.arange(rows))

            # Calcula coordenadas (x,y,w,h) del current_anchor en todas las posiciones del feature map
            A[0, :, :, curr_anchor] = X - anchor_x / 2
            A[1, :, :, curr_anchor] = Y - anchor_y / 2
            A[2, :, :, curr_anchor] = anchor_x
            A[3, :, :, curr_anchor] = anchor_y

            # corrige coordenadas (x,y,w,h) del anchor con deltas (tx,ty,tw,th) predecidos por el modelo RPN
            A[:, :, :, curr_anchor] = apply_regr_rpn(A[:, :, :, curr_anchor], regr)

            # Evita bboxes con altura o anchura menor que 1 (redondea a 1)
            A[2, :, :, curr_anchor] = np.maximum(1, A[2, :, :, curr_anchor])
            A[3, :, :, curr_anchor] = np.maximum(1, A[3, :, :, curr_anchor])

            # Convierte (x, y , w, h) => (x1, y1, x2, y2)
            A[2, :, :, curr_anchor] += A[0, :, :, curr_anchor]
            A[3, :, :, curr_anchor] += A[1, :, :, curr_anchor]

            # Recorta aquellos bboxes que sobresalgan de la imagen (o del feature map)
            A[0, :, :, curr_anchor] = np.maximum(0, A[0, :, :, curr_anchor])
            A[1, :, :, curr_anchor] = np.maximum(0, A[1, :, :, curr_anchor])
            A[2, :, :, curr_anchor] = np.minimum(cols - 1, A[2, :, :, curr_anchor])
            A[3, :, :, curr_anchor] = np.minimum(rows - 1, A[3, :, :, curr_anchor])

            curr_anchor += 1

    # almacena la informacion en forma de listas
    all_boxes = np.reshape(A.transpose((0, 3, 1, 2)), (4, -1)).transpose((1, 0))  # shape => (4050, 4)
    all_probs = out_rpn_cls.transpose((0, 3, 1, 2)).reshape((-1))  # shape => (4050,)

    x1 = all_boxes[:, 0]
    y1 = all_boxes[:, 1]
    x2 = all_boxes[:, 2]
    y2 = all_boxes[:, 3]

    # Elimina bboxes con coordenadas erroneas
    idxs = np.where((x1 - x2 >= 0) | (y1 - y2 >= 0))
    all_boxes = np.delete(all_boxes, idxs, 0)
    all_probs = np.delete(all_probs, idxs, 0)

    # Non_max_suppression. Solo capturamos los bboxes, no necesitamos los scores
    result = non_max_suppression_fast(all_boxes, all_probs, overlap_thresh=overlap_thresh, max_boxes=max_boxes)[0]
    return result


# Redimensiona la imagen al tamano especificado en la configuracion
def format_img_size(img, C):
    #    img_min_side = float(C.im_size)
    #    (height,width,_) = img.shape

    #    if width <= height:
    #        ratio = img_min_side/width
    #        new_height = int(ratio * height)
    #        new_width = int(img_min_side)
    #    else:
    #        ratio = img_min_side/height
    #        new_width = int(ratio * width)
    #        new_height = int(img_min_side)
    #    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    ratio = 1
    return img, ratio


# BGR >> RGB
def format_img_channels(img):
    img = img[:, :, (2, 1, 0)]
    img = np.expand_dims(img, axis=0)
    return img


def format_img(img, C):
    img, ratio = format_img_size(img, C)
    img = format_img_channels(img)
    return img, ratio


# Transforma las coordenadas de los bboxes de la imagen redimensionada a la original
def get_real_coordinates(ratio, x1, y1, x2, y2):
    real_x1 = int(round(x1 // ratio))
    real_y1 = int(round(y1 // ratio))
    real_x2 = int(round(x2 // ratio))
    real_y2 = int(round(y2 // ratio))
    return (real_x1, real_y1, real_x2, real_y2)
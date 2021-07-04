import numpy as np
import math


def non_max_suppression_fast(boxes, probs, overlap_thresh=0.9, max_boxes=300):
    """NMS algorithm to avoid duplicates in the bounding boxes delimiting the same object
    Code extracted from: http://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/

    Step 1: Order probabilities list
    Step 2: Get the highest probability and move it to a new list
    Step 3: Calculate IoU between the bounding box with the previous probability and the other from the list
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


def apply_regress_rpn(X, T):
    """Predicted delta correction for RPN model
    Corrects the coordinates (x,y,w,h) of the anchor based on the deltas (tx,ty,tw,th)

    As stated in the original paper:
    tx=(cx_gt-cx_anchor)/w_anchor, ty=(cy_gt-cy_anchor)/h_anchor, tw=log(w_gt/w_anchor), tw=log(h_gt/h_anchor)
    """

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


def apply_regress_class_final(x, y, w, h, tx, ty, tw, th):
    """Predicted delta correction for the classifier model
    Corrects the coordinates (x,y,w,h) of the anchor based on the deltas (tx,ty,tw,th)

    As stated in the original paper:
    tx=(cx_gt-cx_anchor)/w_anchor, ty=(cy_gt-cy_anchor)/h_anchor, tw=log(w_gt/w_anchor), tw=log(h_gt/h_anchor)
    """

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
    """Define ROI areas based on the scores predictions and deltas of each anchor from RPN model

    Step 1: Calculate ROI bounding boxes. Get coordinates of the anchors of each feature map point

    Step 2: Each anchor is fixed by the predicted deltas by RPN model

    Step 3: Crop those bounding boxes that jut out the image

    Step 4: Apply NMS to the bounding boxes

    :returns: Coordinates of the bounding boxes selected
    """

    # Delta decoding (deltas = deltas*0.25) - i.e. x=(x_gt-x_anc)/(w_anc*var) y w=ln(w_gt/w_anc)/var
    out_rpn_regr = out_rpn_regr / C.std_scaling

    anchor_sizes = C.anchor_box_scales
    anchor_ratios = C.anchor_box_ratios

    assert out_rpn_cls.shape[0] == 1
    (rows, cols) = out_rpn_cls.shape[1:3]

    # A stores the coordinates of the 9 anchors for each feature map point (18x25x9=4050 anchors)
    A = np.zeros((4, out_rpn_cls.shape[1], out_rpn_cls.shape[2], out_rpn_cls.shape[3]))

    curr_anchor = 0
    for anchor_size in anchor_sizes:
        for anchor_ratio in anchor_ratios:
            anchor_x = (anchor_size * anchor_ratio[0]) / C.rpn_stride
            anchor_y = (anchor_size * anchor_ratio[1]) / C.rpn_stride

            # regress store the deltas of current_anchor in all feature map positions
            regress = out_rpn_regr[0, :, :, 4 * curr_anchor:4 * curr_anchor + 4]
            regress = np.transpose(regress, (2, 0, 1))

            # Grid with equal shape to feature map
            X, Y = np.meshgrid(np.arange(cols), np.arange(rows))

            # Calculate coordinates (x,y,w,h) of the current_anchor in all feature map positions
            A[0, :, :, curr_anchor] = X - anchor_x / 2
            A[1, :, :, curr_anchor] = Y - anchor_y / 2
            A[2, :, :, curr_anchor] = anchor_x
            A[3, :, :, curr_anchor] = anchor_y

            # Correct coordinates (x,y,w,h) of the anchor with deltas (tx,ty,tw,th) predicted by RPN model
            A[:, :, :, curr_anchor] = apply_regress_rpn(A[:, :, :, curr_anchor], regress)

            # Avoid bounding boxes with height/width lesser than 1 (rounds to 1)
            A[2, :, :, curr_anchor] = np.maximum(1, A[2, :, :, curr_anchor])
            A[3, :, :, curr_anchor] = np.maximum(1, A[3, :, :, curr_anchor])

            # Transforms (x, y , w, h) => (x1, y1, x2, y2)
            A[2, :, :, curr_anchor] += A[0, :, :, curr_anchor]
            A[3, :, :, curr_anchor] += A[1, :, :, curr_anchor]

            # Crop those bounding boxes that stick out the image or feature map
            A[0, :, :, curr_anchor] = np.maximum(0, A[0, :, :, curr_anchor])
            A[1, :, :, curr_anchor] = np.maximum(0, A[1, :, :, curr_anchor])
            A[2, :, :, curr_anchor] = np.minimum(cols - 1, A[2, :, :, curr_anchor])
            A[3, :, :, curr_anchor] = np.minimum(rows - 1, A[3, :, :, curr_anchor])

            curr_anchor += 1

    all_boxes = np.reshape(A.transpose((0, 3, 1, 2)), (4, -1)).transpose((1, 0))
    all_probabilities = out_rpn_cls.transpose((0, 3, 1, 2)).reshape((-1))

    x1 = all_boxes[:, 0]
    y1 = all_boxes[:, 1]
    x2 = all_boxes[:, 2]
    y2 = all_boxes[:, 3]

    # Remove wrong bounding boxes
    idxs = np.where((x1 - x2 >= 0) | (y1 - y2 >= 0))
    all_boxes = np.delete(all_boxes, idxs, 0)
    all_probabilities = np.delete(all_probabilities, idxs, 0)

    # Non_max_suppression to get only the bounding boxes
    result = non_max_suppression_fast(all_boxes, all_probabilities,
                                      overlap_thresh=overlap_thresh, max_boxes=max_boxes)[0]
    return result


def format_img_channels(img):
    """Convert BGR to RGB"""
    img = img[:, :, (2, 1, 0)]
    img = np.expand_dims(img, axis=0)
    return img


def format_img(img):
    """Formats image color channel and dimension"""
    return format_img_channels(img), 1


def get_real_coordinates(ratio, x1, y1, x2, y2):
    """Transform the bounding boxes coordinates of the rescaled image to the original"""

    real_x1 = int(round(x1 // ratio))
    real_y1 = int(round(y1 // ratio))
    real_x2 = int(round(x2 // ratio))
    real_y2 = int(round(y2 // ratio))
    return real_x1, real_y1, real_x2, real_y2

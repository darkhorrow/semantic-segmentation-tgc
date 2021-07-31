import argparse
import glob
import os.path
import pickle
import time

import cv2
import pandas as pd
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from rbnd_model.base_model import nn_base
from rbnd_model.classifier_model import classifier_layer
from rbnd_model.rpn_model import rpn_layer
from rbnd_model.config import Config
from utils.rbndd_utils import *
from utils.metrics_calculation import calculate_metrics, get_precision, get_recall, get_f1_score


def args_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('-b', '--base_path', help='Base path to the model directory', required=True)
    parser.add_argument('-c', '--config_path', help='Path to pickle file with VGG model config', required=True)
    parser.add_argument('-i', '--images_path', help='Path to the test images to use', required=True)
    parser.add_argument('-o', '--output_path', help='Path to output the predictions performed', required=True)
    parser.add_argument('-t', '--thresh_score', help='Threshold score to display the bounding box', default=0.7)
    parser.add_argument('-v', '--verbose', help='Display informative logs', action='store_true')

    args = parser.parse_args()

    return args.base_path, args.config_path, args.images_path, args.output_path, args.thresh_score, args.verbose


if __name__ == "__main__":
    base_path, config_path, images_path, output_path, bbox_threshold, debug = args_parse()

    with open(config_path, 'rb') as f_in:
        C = pickle.load(f_in)

    # Input layer of VGG model (RGB images)
    img_input = Input(shape=(None, None, 3))

    # Input layer of ROI Pooling model
    roi_input = Input(shape=(C.num_rois, 4))

    # Input layer of classifier model (convolutional feature map (H/stride, W/stride, 512))
    num_features = 512
    feature_map_input = Input(shape=(None, None, num_features))

    # Base network (VGG16)
    shared_layers = nn_base(img_input)

    # RPN model
    num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
    rpn = rpn_layer(shared_layers, num_anchors)

    # Classifier model
    classifier = classifier_layer(feature_map_input, roi_input, C.num_rois, nb_classes=len(C.class_mapping))

    # Build models
    model_rpn = Model(img_input, rpn)
    model_classifier = Model([feature_map_input, roi_input], classifier)

    print('Loading weights from {}'.format(os.path.join(base_path, "model", "model_frcnn_vgg.hdf5")))
    model_rpn.load_weights(os.path.join(base_path, "model", "model_frcnn_vgg.hdf5"), by_name=True)
    model_classifier.load_weights(os.path.join(base_path, "model", "model_frcnn_vgg.hdf5"), by_name=True)

    # Exchange key <-> value pairs
    class_mapping = C.class_mapping
    class_mapping = {v: k for k, v in class_mapping.items()}
    print(class_mapping)

    all_images = []
    classes = {}

    # Store predictions in a Pandas Dataframe
    column_names = ["name", "x_min", "y_min", "x_max", "y_max", "class", "score"]
    predictions = pd.DataFrame(columns=column_names)

    # Store metrics in a Pandas Dataframe
    metrics = pd.DataFrame(columns=['filename', 'precision', 'recall', 'f1_score'])

    # Store separately the global metric result
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    final_metric = pd.DataFrame(columns=['precision', 'recall', 'f1_score'])

    df_bounding_boxes_gt_test = None

    # Writes annotationTest.txt in a Dataframe to draw the ground-truth bounding boxes in the images, if present
    if os.path.exists(os.path.join(images_path, 'annotateTest.csv')):
        df_bounding_boxes_gt_test = pd.read_csv(os.path.join(images_path, 'annotateTest.csv'), sep=",", header=None)
        df_bounding_boxes_gt_test.columns = ["filename", "x_min", "y_min", "x_max", "y_max", "class"]
        df_bounding_boxes_gt_test['difficulty'] = 0
        df_bounding_boxes_gt_test['crowd'] = 0

    for img_file in glob.glob(images_path + "/*.jpg"):
        bib_file = img_file.replace(".jpg", "_bibs.txt")

        if not os.path.exists(bib_file):

            if debug:
                print(f'Processing {img_file}')

            img = cv2.imread(img_file)
            t = time.time()

            # Rescale image and convert BRG to RGB
            X, ratio = format_img(img)

            # Y1: Probability of each anchor to include an object corresponding to each feature map point
            # Y2: Bounding box deltas of each anchor corresponding to each feature map point
            # Delta values are coded with de variance, i.e. x=(x_gt-x_anc)/(w_anc*var) y w=ln(w_gt/w_anc)/var
            # F: Feature map
            [Y1, Y2, F] = model_rpn.predict(X)

            # Fixes the anchors with the delta predictions of the RPN model and chooses bounding boxes according to NMS
            R = rpn_to_roi(Y1, Y2, C, overlap_thresh=0.7)

            if isinstance(R, type(None)):
                print(f'Error processing {img_file}, using next image.')
                continue

            # (x1,y1,x2,y2) => (x,y,w,h)
            R[:, 2] -= R[:, 0]
            R[:, 3] -= R[:, 1]

            bounding_boxes = {}
            probabilities = {}

            for jk in range(R.shape[0] // C.num_rois + 1):
                # Gets the next 4 bounding boxes
                ROIs = np.expand_dims(R[C.num_rois * jk:C.num_rois * (jk + 1), :], axis=0)
                if ROIs.shape[1] == 0:
                    break

                if jk == R.shape[0] // C.num_rois:
                    # Pad R to include 4 ROIs to be fed to the classifier input
                    curr_shape = ROIs.shape
                    target_shape = (curr_shape[0], C.num_rois, curr_shape[2])
                    ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
                    ROIs_padded[:, :curr_shape[1], :] = ROIs
                    ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
                    ROIs = ROIs_padded

                # F: feature maps
                # P_cls (4x2): Score of each ROI and each class (including 'bg')
                # P_regress (4x4): Deltas bounding box (4 values) for each class and ROI
                [P_cls, P_regress] = model_classifier.predict([F, ROIs])

                # Calculate de bounding box coordinates in the original image
                for ii in range(P_cls.shape[1]):
                    # Ignore ROI with (score < bbox_threshold) or (ROI class 'bg')
                    cls_num = np.argmax(P_cls[0, ii, :])
                    if np.max(P_cls[0, ii, :]) < bbox_threshold or cls_num == (P_cls.shape[2] - 1):
                        continue

                    cls_name = class_mapping[cls_num]
                    if cls_name not in bounding_boxes:
                        bounding_boxes[cls_name] = []
                        probabilities[cls_name] = []

                    (x, y, w, h) = ROIs[0, ii, :]

                    try:
                        # Extract deltas
                        (tx, ty, tw, th) = P_regress[0, ii, 4 * cls_num:4 * (cls_num + 1)]
                        tx /= C.classifier_regr_std[0]
                        ty /= C.classifier_regr_std[1]
                        tw /= C.classifier_regr_std[2]
                        th /= C.classifier_regr_std[3]

                        # ROI bounding box correction
                        x, y, w, h = apply_regress_class_final(x, y, w, h, tx, ty, tw, th)
                    except Exception as error:
                        print(error)

                    # Store results of bounding boxes and scores
                    bounding_boxes[cls_name].append(
                        [C.rpn_stride * x, C.rpn_stride * y, C.rpn_stride * (x + w), C.rpn_stride * (y + h)])
                    probabilities[cls_name].append(np.max(P_cls[0, ii, :]))

            all_detects = []

            with open(bib_file, "w") as fid:
                # Draw true ground bounding boxes
                if df_bounding_boxes_gt_test is not None:
                    gt_bounding_boxes = df_bounding_boxes_gt_test[df_bounding_boxes_gt_test['filename'] == img_file]
                    for row in gt_bounding_boxes.itertuples():
                        cv2.rectangle(img, (row.x_min, row.y_min), (row.x_max, row.y_max), (0, 255, 0), 2)
                # No detections performed
                if len(bounding_boxes) == 0:
                    print('No detections for this image')
                    # Calculate mAP
                    if df_bounding_boxes_gt_test is not None:
                        gt_bounding_boxes = df_bounding_boxes_gt_test[df_bounding_boxes_gt_test['filename'] == img_file]
                        gt_bounding_boxes = gt_bounding_boxes.drop(['filename'], axis=1)
                        gt_bounding_boxes['class'] = 0
                        gt_bounding_boxes = gt_bounding_boxes.to_numpy()

                        print(gt_bounding_boxes)

                        predictions_df = predictions[predictions['name'] == img_file]
                        predictions_df = predictions_df.drop('name', axis=1)
                        predictions_df['class'] = 0
                        predicted_bounding_boxes = predictions_df.to_numpy()
                        print(predicted_bounding_boxes)

                        (tp, fp, fn), (p, r, f1s) = calculate_metrics(gt_bounding_boxes[:, 0:4],
                                                                      predicted_bounding_boxes[:, 0:4])

                        if debug:
                            print('Found:')
                            print(f'TP: {tp}\tFP: {fp}\tFN: {fn}\tPrecision: {p}\tRecall: {r}\tF1-score: {f1s}')

                        true_positives += tp
                        false_positives += fp
                        false_negatives += fn

                        print(f'Global true positives: {true_positives}\tfalse positives: {false_positives}\t'
                              f'false negatives: {false_negatives}')

                        metrics = metrics.append(
                            {
                                "filename": img_file,
                                "precision": p,
                                "recall": r,
                                "f1_score": f1s
                            },
                            ignore_index=True
                        )

                # In case detections exist
                for key in bounding_boxes:
                    bbox = np.array(bounding_boxes[key])

                    new_boxes, new_probabilities = non_max_suppression_fast(bbox, np.array(probabilities[key]), 0.2)

                    for jk in range(new_boxes.shape[0]):
                        (x1, y1, x2, y2) = new_boxes[jk, :]

                        # Calculate coordinates in the original image and draw the detected bounding box
                        (real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)
                        cv2.rectangle(img, (real_x1, real_y1), (real_x2, real_y2), (0, 0, 255), 2)

                        # Detection data to txt file coordinates and confidence
                        fid.write("%d %d %d %d %d\n" %
                                  (real_x1, real_y1, real_x2, real_y2, int(100 * new_probabilities[jk])))

                        textLabel = '{}: {}'.format("Score", int(100 * new_probabilities[jk]))
                        all_detects.append((key, 100 * new_probabilities[jk]))

                        (ret_val, baseLine) = cv2.getTextSize(textLabel, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)
                        textOrg = (real_x1, real_y1)
                        xxx1 = textOrg[0] - 0
                        yyy1 = textOrg[1] + baseLine - 0
                        xxx2 = textOrg[0] + ret_val[0] + 0
                        yyy2 = textOrg[1] - ret_val[1] - 0
                        if xxx1 < 0 or yyy1 < 0 or xxx2 < 0 or yyy2 < 0:
                            textOrg = (real_x2, real_y2)
                            xxx1 = textOrg[0] - ret_val[0] - 0
                            yyy1 = textOrg[1] + ret_val[1] + 0
                            xxx2 = textOrg[0] + 0
                            yyy2 = textOrg[1] - baseLine + 0
                            textOrg = (xxx1, yyy1 - baseLine)

                        cv2.rectangle(img, (xxx1, yyy1), (xxx2, yyy2), (0, 0, 0), 1)
                        cv2.rectangle(img, (xxx1, yyy1), (xxx2, yyy2), (255, 255, 255), -1)
                        cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1)

                        predictions = predictions.append(
                            {
                                "name": img_file,
                                "x_min": real_x1,
                                "y_min": real_y1,
                                "x_max": real_x2,
                                "y_max": real_y2,
                                "class": key,
                                "score": int(100 * new_probabilities[jk])
                            },
                            ignore_index=True
                        )

                    # Calculate mAP
                    if df_bounding_boxes_gt_test is not None:
                        gt_bounding_boxes = df_bounding_boxes_gt_test[df_bounding_boxes_gt_test['filename'] == img_file]
                        gt_bounding_boxes = gt_bounding_boxes.drop(['filename'], axis=1)
                        gt_bounding_boxes['class'] = 0
                        gt_bounding_boxes = gt_bounding_boxes.to_numpy()

                        predictions_df = predictions[predictions['name'] == img_file]
                        predictions_df = predictions_df.drop('name', axis=1)
                        predictions_df['class'] = 0
                        predicted_bounding_boxes = predictions_df.to_numpy()

                        (tp, fp, fn), (p, r, f1s) = calculate_metrics(gt_bounding_boxes[:, 0:4],
                                                                      predicted_bounding_boxes[:, 0:4])

                        if debug:
                            print('Found:')
                            print(f'TP: {tp}\tFP: {fp}\tFN: {fn}\tPrecision: {p}\tRecall: {r}\tF1-score: {f1s}')

                        true_positives += tp
                        false_positives += fp
                        false_negatives += fn

                        print(f'Global true positives: {true_positives}\tfalse positives: {false_positives}\t'
                              f'false negatives: {false_negatives}')

                        metrics = metrics.append(
                            {
                                "filename": img_file,
                                "precision": p,
                                "recall": r,
                                "f1_score": f1s
                            },
                            ignore_index=True
                        )

                    # Store the image with the detection
                    print(f'Saved in {os.path.join(output_path, os.path.basename(img_file))}')
                    cv2.imwrite(os.path.join(output_path, os.path.basename(img_file)), img)

                if debug:
                    print("Elapsed time: {:.3f}".format(time.time() - t))

    final_metric = final_metric.append(
        {
            "precision": get_precision(true_positives, false_positives),
            "recall": get_recall(true_positives, false_negatives),
            "f1_score": get_f1_score(true_positives, false_positives, false_negatives)
        },
        ignore_index=True
    )

    predictions.to_csv(os.path.join(output_path, 'predictions.csv'))
    metrics.to_csv(os.path.join(output_path, 'metrics.csv'))
    final_metric.to_csv(os.path.join(output_path, 'final_metrics.csv'))

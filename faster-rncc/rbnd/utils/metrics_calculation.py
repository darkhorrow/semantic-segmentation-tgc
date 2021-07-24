import numpy as np
from shapely.geometry import Polygon


def get_precision(true_positives, false_positives):
    return true_positives / (true_positives + false_positives)


def get_recall(true_positives, false_negatives):
    return true_positives / (true_positives + false_negatives)


def get_f1_score(true_positives, false_positives, false_negatives):
    precision = get_precision(true_positives, false_positives)
    recall = get_recall(true_positives, false_negatives)
    return 2 * (precision * recall) / (precision + recall) if precision > 0 or recall > 0 else 0


def get_f1_score_with_pr(precision, recall):
    return 2 * (precision * recall) / (precision + recall) if precision > 0 or recall > 0 else 0


def set_diff(a, b):
    """Set difference to get a elements not present in b"""
    c = np.array([])
    for element in a:
        if not exist_element(element, b):
            c = np.append(c, element)
    return c


def exist_element(element, collection):
    return np.isin(element, collection).all()


def iou(ground_truth, detection, threshold=0.5):
    x_max = max(ground_truth[0], detection[0])
    y_max = max(ground_truth[1], detection[1])
    x_min = min(ground_truth[2], detection[2])
    y_min = min(ground_truth[3], detection[3])

    intersection_area = max(0, x_min - x_max) * max(0, y_min - y_max)

    ground_truth_area = (ground_truth[2] - ground_truth[0]) * (ground_truth[3] - ground_truth[1])
    detection_area = (detection[2] - detection[0]) * (detection[3] - detection[1])

    union_area = ground_truth_area + detection_area - intersection_area

    result = intersection_area / union_area

    return result >= threshold


def calculate_metrics(ground_truths, detections, iou_threshold=0.5):
    true_positives = 0
    false_positive = 0
    false_negative = 0

    gt_done = np.array([])
    predict_done = np.array([])

    if ground_truths is None or len(ground_truths) == 0:
        false_positive += len(detections)
        precision = get_precision(true_positives, false_positive)
        recall = get_recall(true_positives, false_negative)
        f_score = get_f1_score_with_pr(precision, recall)
        return precision, recall, f_score

    if detections is None or len(detections) == 0:
        false_negative += len(ground_truths)
        precision = get_precision(true_positives, false_positive)
        recall = get_recall(true_positives, false_negative)
        f_score = get_f1_score_with_pr(precision, recall)
        return precision, recall, f_score

    for ground_truth in ground_truths:
        for detection in detections:
            if not exist_element(detection, predict_done):
                if iou(ground_truth, detection, iou_threshold):
                    true_positives += 1
                    predict_done = np.append(predict_done, detection)
                    gt_done = np.append(gt_done, ground_truth)
                    break

    false_positive_detections = set_diff(detections, predict_done)
    false_positive = int(len(false_positive_detections) / 4)

    false_negative_detections = set_diff(ground_truths, gt_done)
    false_negative = int(len(false_negative_detections) / 4)

    precision = get_precision(true_positives, false_positive)
    recall = get_recall(true_positives, false_negative)
    f_score = get_f1_score_with_pr(precision, recall)

    return (true_positives, false_positive, false_negative), (precision, recall, f_score)

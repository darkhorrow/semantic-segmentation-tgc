import numpy as np


def iou(ground_truth, detection, threshold=0.5):
    x_max = max(ground_truth[0], detection[0])
    y_max = max(ground_truth[1], detection[1])
    x_min = min(ground_truth[2], detection[2])
    y_min = min(ground_truth[3], detection[3])

    intersection_area = max(0, x_min - x_max + 1) * max(0, y_min - y_max + 1)

    ground_truth_area = (ground_truth[2] - ground_truth[0] + 1) * (ground_truth[3] - ground_truth[1] + 1)
    detection_area = (detection[2] - detection[0] + 1) * (detection[3] - detection[1] + 1)

    union_area = ground_truth_area + detection_area - intersection_area

    result = intersection_area / union_area

    return result >= threshold


def calculate_metrics(ground_truths, detections, iou_threshold=0.5):
    true_positives = 0
    false_positive = 0
    false_negative = 0

    ground_truths_detected = np.array([])

    if ground_truths is None or len(ground_truths) == 0:
        false_positive += len(detections)
        print('None ground truths')
        print(true_positives, false_positive, false_negative)
        return

    if detections is None or len(detections) == 0:
        false_negative += len(ground_truths)
        print('None detections')
        print(true_positives, false_positive, false_negative)
        return

    for detection in detections:
        for ground_truth in ground_truths:
            if not np.isin(ground_truth, ground_truths_detected).all() and iou(ground_truth, detection, iou_threshold):
                true_positives += 1
                np.append(ground_truths_detected, ground_truth)
            else:
                false_positive += 1

    print(true_positives, false_positive, false_negative)

    precision = true_positives/(true_positives + false_positive)
    recall = true_positives/(true_positives + false_negative)
    f_score = 2 * (precision * recall)/(precision + recall)

    print(precision, recall, f_score)
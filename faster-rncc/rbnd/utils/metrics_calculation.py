def iou(ground_truths, detection, threshold=0.5):
    results = []
    for ground_truth in ground_truths:
        x_max = max(ground_truth[0], detection[0])
        y_max = max(ground_truth[1], detection[1])
        x_min = min(ground_truth[2], detection[2])
        y_min = min(ground_truth[3], detection[3])

        intersection_area = max(0, x_min - x_max + 1) * max(0, y_min - y_max + 1)

        ground_truth_area = (ground_truth[2] - ground_truth[0] + 1) * (ground_truth[3] - ground_truth[1] + 1)
        detection_area = (detection[2] - detection[0] + 1) * (detection[3] - detection[1] + 1)

        union_area = ground_truth_area + detection_area - intersection_area

        results.append(intersection_area / union_area)
    return max(results) >= threshold


def calculate_metrics(ground_truths, detection, iou_threshold=0.5):
    # True positive: there is an annotation for the prediction
    if iou(ground_truths, detection, iou_threshold):
        pass
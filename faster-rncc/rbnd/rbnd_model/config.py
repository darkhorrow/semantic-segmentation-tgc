import math


class Config:
    """Configuration class that stores the parameters to be used in the detector"""

    def __init__(self):
        # Anchors sizes
        self.anchor_box_scales = [32, 64, 128]

        # Anchors ratios
        self.anchor_box_ratios = [[1, 1], [1. / math.sqrt(2), 2. / math.sqrt(2)],
                                  [2. / math.sqrt(2), 1. / math.sqrt(2)]]

        # Number of ROI areas processed simultaneously
        self.num_rois = 4

        # RPN model stride (base model VGG16)
        self.rpn_stride = 16

        # Scaling the std
        self.std_scaling = 4.0
        self.classifier_regr_std = [8.0, 8.0, 4.0, 4.0]

        # RPN model thresholds
        self.rpn_min_overlap = 0.3
        self.rpn_max_overlap = 0.7

        # Classifier thresholds
        self.classifier_min_overlap = 0.1
        self.classifier_max_overlap = 0.5

        # Class codifications
        self.class_mapping = None
        self.model_path = None

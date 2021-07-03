from .roi_pooling_conv import RoiPoolingConv
from tensorflow.keras.layers import Flatten, Dense, Dropout, TimeDistributed


def classifier_layer(base_layers, input_rois, num_rois, nb_classes):
    """Classifier model"""

    pooling_regions = 7

    # TimeDistributed layers are used to process ROI areas independently
    # It is used the number of ROI's + an extra dimension (num_rois)
    # out_roi_pool is a list of 4 ROI (7x7x512)
    # out_roi_pool.shape = (1, num_rois, pool_size, pool_size, channels)
    out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([base_layers, input_rois])

    # Flatten out_roi_pool and connect to 2 Fully-Connected and 2 Dropout layers
    out = TimeDistributed(Flatten(name='flatten'))(out_roi_pool)
    out = TimeDistributed(Dense(4096, activation='relu', name='fc1'))(out)
    out = TimeDistributed(Dropout(0.5))(out)
    out = TimeDistributed(Dense(4096, activation='relu', name='fc2'))(out)
    out = TimeDistributed(Dropout(0.5))(out)

    # out_class: Class prediction
    out_class = TimeDistributed(Dense(nb_classes, activation='softmax', kernel_initializer='zero'),
                                name='dense_class_{}'.format(nb_classes))(out)

    # out_regress: Coordinates predictions of the bounding boxes
    out_regress = TimeDistributed(Dense(4 * (nb_classes - 1), activation='linear', kernel_initializer='zero'),
                                  name='dense_regress_{}'.format(nb_classes))(out)

    return [out_class, out_regress]

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer


class RoiPoolingConv(Layer):
    """ROI Pooling Convolutional Layer definition"""

    def __init__(self, pool_size, num_rois, **kwargs):
        self.dim_ordering = K.image_data_format()
        self.pool_size = pool_size
        self.num_rois = num_rois
        self.nb_channels = None
        super(RoiPoolingConv, self).__init__(**kwargs)

    def build(self, input_shape):
        self.nb_channels = input_shape[0][3]

    def compute_output_shape(self, input_shape):
        return None, self.num_rois, self.pool_size, self.pool_size, self.nb_channels

    def call(self, x, mask=None):
        image = x[0]
        rois = x[1]
        outputs = []

        for roi_idx in range(self.num_rois):
            x = K.cast(rois[0, roi_idx, 0], 'int32')
            y = K.cast(rois[0, roi_idx, 1], 'int32')
            w = K.cast(rois[0, roi_idx, 2], 'int32')
            h = K.cast(rois[0, roi_idx, 3], 'int32')

            region = image[:, y:y + h, x:x + w, :]

            # ROI pooling is not possible to be calculated on regions with lesser area than 7x7, proceed to reshape
            rs = tf.image.resize(region, (self.pool_size, self.pool_size))
            outputs.append(rs)

        final_output = K.concatenate(outputs, axis=0)

        # Reshape to (1, num_rois, pool_size, pool_size, nb_channels)
        final_output = K.reshape(final_output, (1, self.num_rois, self.pool_size, self.pool_size, self.nb_channels))
        final_output = K.permute_dimensions(final_output, (0, 1, 2, 3, 4))

        return final_output

    def get_config(self):
        config = {'pool_size': self.pool_size, 'num_rois': self.num_rois}
        base_config = super(RoiPoolingConv, self).get_config()
        result = list(base_config.items()) + list(config.items())
        return dict(result)

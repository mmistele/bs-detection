from keras import backend as K
from keras.layers import Layer
import numpy as np
import tensorflow as tf

# Credit to Jacoxu in https://stackoverflow.com/questions/39510809/mean-or-max-pooling-with-masking-support-in-keras
class MeanPool(Layer):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(MeanPool, self).__init__(**kwargs)

    def compute_mask(self, input, input_mask=None):
        return None
    
    def call(self, x, mask=None):
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            mask = tf.expand_dims(mask, -1)
            x = x * mask
        avg = K.sum(x, axis=1) / K.sum(mask, axis=1)
        return avg
    
    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]



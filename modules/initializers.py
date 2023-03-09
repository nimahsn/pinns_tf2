import tensorflow as tf
import numpy as np

class NtkNormal(tf.keras.initializers.Initializer):
    def __init__(self, seed) -> None:
        super().__init__()
        self.seed = seed
        self.rangen = tf.random.Generator.from_seed(self.seed)

    def __call__(self, shape, dtype=tf.float32):
        n_in = shape[0]
        n_out = shape[1]
        std = 1. / np.sqrt(n_in)
        return self.rangen.normal(shape, dtype=dtype) * std
        
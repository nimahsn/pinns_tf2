import argparse
import os
import sys
import time
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'
parser = argparse.ArgumentParser()
parser.add_argument('--width', type=int, default=32, help='width of network')
parser.add_argument('--depth', type=int, default=4, help='depth of network')
# epochs 
parser.add_argument('--epochs', type=int, default=50000, help='number of epochs')
# activation
parser.add_argument('--activation', type=str, default='cos', help='activation function')
# wave frequency
parser.add_argument('--c', type=float, default=1.0, help='speed of the wave')
# number of samples
parser.add_argument('--n', type=int, default=512, help='number of samples')
# seed
parser.add_argument('--seed', type=int, default=11, help='seed for random number generator')
# memory limit
parser.add_argument('--mem_limit', type=int, default=4096, help='memory limit for GPU')
#
parser.add_argument('--init_c', type=float, default=3.0, help='initialization c')


args = parser.parse_args()
width = args.width
depth = args.depth
epochs = args.epochs
c = args.c
activation = args.activation
activation_st = activation
n = args.n
seed = args.seed
mem_limit = args.mem_limit
init_c = args.init_c

from modules.utils import save_history_csv
from modules.models import create_dense_model, WavePinn
from modules.data import simulate_wave
import tensorflow as tf
import numpy as np

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.\
                                                                      VirtualDeviceConfiguration(memory_limit=5000)])
  except RuntimeError as e:
    print(e)

if activation == 'cos':
    activation = tf.cos
elif activation == 'sin':
    activation = tf.sin
elif activation == 'tanh':
    activation = tf.keras.activations.tanh

class UniformSine(tf.keras.initializers.Initializer):
    def __init__(self, c=6.0, seed=42):
        self.c = c
        self.gen = tf.random.Generator.from_seed(seed)

    def __call__(self, shape, dtype=None):
        n_units = shape[0]
        return self.gen.uniform(shape, minval=-tf.math.sqrt(self.c / n_units), maxval=tf.math.sqrt(self.c / n_units),  dtype=dtype)

    def get_config(self):
        return {'c': self.c}

#https://personal.math.ubc.ca/~feldman/m267/separation.pdf
c = c
length = 1.0
n_samples = n

def f_u(tx):
    t = tx[:, 0:1]
    x = tx[:, 1:2]
    return tf.sin(5 * np.pi * x) * tf.cos(5 * c * np.pi * t) + \
        2*tf.sin(7 * np.pi * x) * tf.cos(7 * c * np.pi * t)

def f_u_init(tx):
    x = tx[:, 1:2]
    return tf.sin(5 * np.pi * x) + 2*tf.sin(7 * np.pi * x)

def f_du_dt(tx):
    return tf.zeros_like(tx[:, 0:1])

def f_u_bnd(tx):
    return tf.zeros_like(tx[:, 1:2])

(tx_samples, residual), (tx_init, u_init, du_dt_init), (tx_bndry, u_bndry) = \
    simulate_wave(n_samples, f_u_init, f_du_dt, f_u_bnd)

inputs = [tx_samples, tx_init, tx_bndry]
outputs = [f_u(tx_samples), residual, u_init, du_dt_init, u_bndry]

mean = np.mean(tx_samples, axis=0)
std = np.std(tx_samples, axis=0)

backbone = create_dense_model([tf.keras.layers.Normalization(mean=mean, variance=std**2)] + [width]*depth, activation=activation,
                              initializer=tf.keras.initializers.GlorotNormal(seed=seed), n_inputs=2, n_outputs=1)
model = WavePinn(backbone, c)
scheduler = tf.keras.optimizers.schedules.ExponentialDecay(1e-3, 1000, 0.95)
optimizer = tf.keras.optimizers.Adam(scheduler)
model.compile(optimizer=optimizer)

name = f'wave_backbone' + '_width_' + str(width) + '_depth_' + str(depth) + \
    f'_activation_{activation_st}' + '_epochs_' + str(epochs) + '_c_' + str(c) + \
        '_n_' + str(n) + '_seed_' + str(seed)
print(name)

hist = model.fit_custom(inputs, outputs, epochs=epochs, print_every=1000)


backbone.save('./new_experiments/models/' + name)
save_history_csv(hist, name, './new_experiments/history')

#save weights
# backbone.save_weights('./experiments/weights/' + name + '.h5')


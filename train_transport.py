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
# wave frequency
parser.add_argument('--beta', type=int, default=30, help='frequency of the sin wave')
# number of samples
parser.add_argument('--nc', type=int, default=500, help='number of colloc samples')
parser.add_argument('--nb', type=int, default=200, help='number of boundary samples')
parser.add_argument('--ni', type=int, default=200, help='number of samples for initial condition')
# seed
parser.add_argument('--seed', type=int, default=11, help='seed for random number generator')
# memory limit
parser.add_argument('--mem_limit', type=int, default=4096, help='memory limit for GPU')
parser.add_argument('--activation', type=str, default='cos', help='activation function')
parser.add_argument('--init_c', type=float, default=3.0, help='initialization c')


args = parser.parse_args()
width = args.width
depth = args.depth
epochs = args.epochs
beta = args.beta
nc = args.nc
nb = args.nb
ni = args.ni
seed = args.seed
mem_limit = args.mem_limit
init_c = args.init_c

activation = args.activation

import tensorflow as tf
from tensorflow import keras
from modules.models import TransportEquation, create_dense_model
from modules.data import simulate_transport
from modules.utils import PrintLossCallback, save_history_csv
from data.transport import convection_diffusion
import numpy as np

activation_st = activation
if activation == 'cos':
    activation = tf.cos
elif activation == 'sin':
    activation = tf.sin
elif activation == 'tanh':
    activation = tf.keras.activations.tanh
elif activation == 'softplus':
    activation = tf.keras.activations.softplus

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5000)])
  except RuntimeError as e:
    print(e)

class UniformSine(tf.keras.initializers.Initializer):
    def __init__(self, c=6.0, seed=42):
        self.c = c
        self.gen = tf.random.Generator.from_seed(seed)

    def __call__(self, shape, dtype=None):
        n_units = shape[0]
        return self.gen.uniform(shape, minval=-tf.math.sqrt(self.c / n_units), maxval=tf.math.sqrt(self.c / n_units),  dtype=dtype)

    def get_config(self):
        return {'c': self.c}

(tx_samples, u_samples, samples_residuals), (tx_init, u_init), (tx_bnd_start, tx_bnd_end, u_boundary), (X, T, U) = simulate_transport(
    nc, ni, nb, convection_diffusion, beta, time_steps=200, x_steps=256
)

inputs = [tx_samples, tx_init, tx_bnd_start, tx_bnd_end]
outputs = [u_samples, samples_residuals, u_init]

mean = np.mean(tx_samples, axis=0)
std = np.std(tx_samples, axis=0)

# backbone = create_dense_model([tf.keras.layers.Normalization(mean=mean, variance=std**2)] + [width]*depth, tf.cos, 
#                               tf.keras.initializers.GlorotNormal(seed=seed), 2, 1)
backbone = create_dense_model([tf.keras.layers.Normalization(mean=mean, variance=std**2)] + [width]*depth, activation=activation,
                              initializer=tf.keras.initializers.GlorotNormal(seed=seed), n_inputs=2, n_outputs=1)
print(backbone.summary())
model = TransportEquation(backbone, beta)
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(1e-3, 1000, 0.95)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
model.compile(optimizer)
hist = model.fit_custom(inputs, outputs, epochs=epochs, print_every=1000)

# name = 'poisson_backbone' + '_width_' + str(width) + '_depth_' + str(depth) + '_epochs_' + str(epochs) + '_freq_' + str(a) + '_samples_' + str(samples) + '_seed_' + str(seed)
name = f'transport_backbone' + '_width_' + str(width) + '_depth_' + str(depth) + '_activation_' + activation_st + '_epochs_' + str(epochs) +\
    '_beta_' + str(beta) + '_nc_' + str(nc) + '_nb_' + str(nb) + '_ni_' + str(ni) + '_seed_' + str(seed)
# save model
backbone.save('./new_experiments/models/' + name)
# save history csv
save_history_csv(hist, name, './new_experiments/history')

import argparse
from mimetypes import init

parser = argparse.ArgumentParser()
parser.add_argument('--width', type=int, default=32, help='width of network')
parser.add_argument('--depth', type=int, default=4, help='depth of network')
# epochs 
parser.add_argument('--epochs', type=int, default=50000, help='number of epochs')
# number of samples
parser.add_argument('--nc', type=int, default=250, help='number of colloc samples')
parser.add_argument('--ni', type=int, default=200, help='number of ic samples')
parser.add_argument('--nb', type=int, default=200, help='number of bc samples')
# seed
parser.add_argument('--seed', type=int, default=11, help='seed for random number generator')

parser.add_argument('--activation', type=str, default='tanh', help='activation function')

parser.add_argument('--init_c', type=float, default=6.0, help='initialization c')

args = parser.parse_args()
width = args.width
depth = args.depth
epochs = args.epochs
seed = args.seed
n_samples = args.nc
n_init = args.ni
n_bcs = args.nb
activation = args.activation
activation_st = activation
init_c = args.init_c

import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5000)])
  except RuntimeError as e:
    print(e)


from tensorflow import keras
from modules.models import KleinGordonEquation, create_dense_model
from modules.utils import save_history_csv
from modules.data import simulate_klein_gordon
import numpy as np
from modules.initializers import UniformSine

if activation == 'cos':
    activation = tf.cos
elif activation == 'sin':
    activation = tf.sin
elif activation == 'tanh':
    activation = tf.keras.activations.tanh

def f_u_exact(tx):
    """Exact solution of the Klein-Gordon equation."""
    return tx[:, 1:2] * tf.cos(5 * np.pi * tx[:, 0:1]) + (tx[:, 0:1] * tx[:, 1:2]) ** 3

def f_rhs(tx):
    """Right-hand side of the Klein-Gordon equation."""
    u = f_u_exact(tx)
    u_tt = -25 * np.pi ** 2 * tx[:, 1:2] * tf.cos(5 * np.pi * tx[:, 0:1]) + 6 * tx[:, 1:2] ** 3 * tx[:, 0:1]
    u_xx = 6 * tx[:, 0:1] ** 3 * tx[:, 1:2]
    return u_tt + -1 * u_xx + 1 * (u ** 3)

def f_ut_exact(tx):
    """Exact time derivative of the Klein-Gordon equation."""
    return -5 * np.pi * tx[:, 1:2] * tf.sin(5 * np.pi * tx[:, 0:1]) + (3.0 * tx[:, 0:1] ** 2) * (tx[:, 1:2] ** 3)

(tx_colloc, y_res), (tx_init, u_init, ut_init), (tx_bnd, u_bnd) = simulate_klein_gordon(
    n_colloc=n_samples, n_init=n_init, n_bnd=n_bcs, rhs_function=f_rhs, init_function=f_u_exact, \
        bnd_function=f_u_exact, init_ut_function=f_ut_exact)
inputs = [tx_colloc, tx_init, tx_bnd]
outputs = [f_u_exact(tx_colloc), y_res, u_init, ut_init, u_bnd]

mean = np.mean(tx_colloc, axis=0)
std = np.std(tx_colloc, axis=0)

name = f'klein(SineInit{init_c})_backbone' + '_width_' + str(width) + '_depth_' + str(depth) + '_activation_' + activation_st + '_epochs_' + str(epochs) + '_nc_' + str(n_samples) + '_ni_' + str(n_init) + '_nb_' + str(n_bcs) + '_seed_' + str(seed)
print(name)

backbone = create_dense_model([tf.keras.layers.Normalization(mean=mean, variance=std**2)] + [width]*depth, activation=activation,
                              initializer=UniformSine(seed=seed, c=init_c), n_inputs=2, n_outputs=1)

model = KleinGordonEquation(backbone)
scheduler = tf.keras.optimizers.schedules.ExponentialDecay(1e-3, decay_steps=1000, decay_rate=0.95)
optimizer = tf.keras.optimizers.Adam(learning_rate=scheduler)
model.compile(optimizer=optimizer)
history = model.fit_custom(inputs, outputs, epochs, 1000)

# name = 'reacdiff_backbone' + '_width_' + str(width) + '_depth_' + str(depth) + '_epochs_' + str(epochs) + '_nu_' + str(nu) + '_rho_' + str(rho) + '_nc_' + str(n_samples) + '_ni_' + str(n_init) + '_nb_' + str(n_bcs) + '_seed_' + str(seed)
backbone.save('./new_experiments/models/' + name)
# save history csv
save_history_csv(history, name, './new_experiments/history')

# save weights
# backbone.save_weights('./experiments/weights/' + name + '.h5')

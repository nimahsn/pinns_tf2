import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--width', type=int, default=32, help='width of network')
parser.add_argument('--depth', type=int, default=4, help='depth of network')
# epochs 
parser.add_argument('--epochs', type=int, default=50000, help='number of epochs')
# wave frequency
parser.add_argument('--rho', type=float, default=5.0, help='rho')
parser.add_argument('--nu', type=float, default=3.0, help='nu')
# number of samples
parser.add_argument('--nc', type=int, default=1024, help='number of colloc samples')
parser.add_argument('--ni', type=int, default=256, help='number of ic samples')
parser.add_argument('--nb', type=int, default=256, help='number of bc samples')

# seed
parser.add_argument('--seed', type=int, default=11, help='seed for random number generator')

parser.add_argument('--activation', type=str, default='tanh', help='activation function')

args = parser.parse_args()
width = args.width
depth = args.depth
epochs = args.epochs
seed = args.seed
nu = args.nu
rho = args.rho
n_samples = args.nc
n_init = args.ni
n_bcs = args.nb
activation = args.activation

import numpy as np
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5096)])
  except RuntimeError as e:
    print(e)


from modules.models import BurgersPinn, create_dense_model
from modules.data import simulate_burgers
from modules.plots import plot_training_loss_linlog, plot_burgers_model
from modules.utils import load_mat_data
import tensorflow as tf
from modules.utils import PrintLossCallback, save_history_csv

activation_st = activation
if activation == 'cos':
    activation = tf.cos
elif activation == 'sin':
    activation = tf.sin
elif activation == 'tanh':
    activation = tf.keras.activations.tanh


data = load_mat_data('./data/burgers_shock.mat')
x = data['x']
t = data['t']
# create a meshgrid
X, T = np.meshgrid(x, t)
# flatten the meshgrid
tx_train = np.hstack((T.flatten()[:,None], X.flatten()[:,None]))
u_train = data['usol'].T.flatten()[:,None]
# convert to tf
tx_train = tf.convert_to_tensor(tx_train, dtype=tf.float32)
u_train = tf.convert_to_tensor(u_train, dtype=tf.float32)

idx = np.random.choice(tx_train.shape[0], n_samples, replace=False)
tx_train = tf.gather(tx_train, idx)
u_train = tf.gather(u_train, idx)

(_, y_samples), (tx_init, u_init), (tx_boundary, u_boundary) = simulate_burgers(n_samples, n_init=n_init, n_bndry=n_bcs)
inputs = [tx_train, tx_init, tx_boundary]
outputs = [u_train, y_samples, u_init, u_boundary]
inputs = [tx_train, tx_init, tx_boundary]
outputs = [u_train, y_samples, u_init, u_boundary]

mean = tf.reduce_mean(tx_train, axis=0)
std = tf.math.reduce_std(tx_train, axis=0)
name = f'burgers_backbone' + '_width_' + str(width) + '_depth_' + str(depth) + '_activation_' + activation_st + '_epochs_' + str(epochs) + '_nc_' + str(n_samples) + '_ni_' + str(n_init) + '_nb_' + str(n_bcs) + '_seed_' + str(seed)
print(name)
init = tf.keras.initializers.GlorotNormal(seed=seed)
backbone = create_dense_model([tf.keras.layers.Normalization(mean=mean, variance=std**2)] + [width]*depth, activation, init, n_inputs = 2, n_outputs = 1)
pinn = BurgersPinn(backbone, 0.01/np.pi)
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(1e-3, decay_steps=1000, decay_rate=0.95)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
pinn.compile(optimizer=optimizer)
history = pinn.fit_custom(inputs, outputs, epochs, 1000)

# name = 'poisson_backbone' + '_width_' + str(width) + '_depth_' + str(depth) + '_epochs_' + str(epochs) + '_freq_' + str(a) + '_samples_' + str(samples) + '_seed_' + str(seed)
# save history csv
backbone.save('./rebuttal/models/' + name)

save_history_csv(history, name, './rebuttal/history')


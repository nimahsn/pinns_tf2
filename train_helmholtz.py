import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--width', type=int, default=32, help='width of network')
parser.add_argument('--depth', type=int, default=4, help='depth of network')
# epochs 
parser.add_argument('--epochs', type=int, default=50000, help='number of epochs')
# number of samples
parser.add_argument('--nc', type=int, default=256, help='number of colloc samples')
# seed
parser.add_argument('--seed', type=int, default=42, help='seed for random number generator')
# memory limit
parser.add_argument('--mem_limit', type=int, default=4096, help='memory limit for GPU')
parser.add_argument('--activation', type=str, default='cos', help='activation function')
parser.add_argument('--initializer', type=str, default='glorot', help='initializer')
parser.add_argument('--init_c', type=float, default=3.0, help='initialization c')


args = parser.parse_args()
width = args.width
depth = args.depth
epochs = args.epochs
nc = args.nc
seed = args.seed
mem_limit = args.mem_limit
activation = args.activation
initializer = args.initializer
init_c = args.init_c

import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.\
                                                                      VirtualDeviceConfiguration(memory_limit=5000)])
  except RuntimeError as e:
    print(e)

import tensorflow.keras as keras
from modules.models import Helmholtz2DPinn, create_dense_model
from modules.utils import save_history_csv
from modules.initializers import UniformSine
import numpy as np

activation_st = activation
if activation == 'cos':
    activation = tf.cos
elif activation == 'sin':
    activation = tf.sin
elif activation == 'tanh':
    activation = tf.keras.activations.tanh

initializer_st = initializer
if initializer == 'glorot':
    initializer = tf.keras.initializers.GlorotNormal(seed=seed)
elif initializer == 'sine':
    initializer = UniformSine(c=init_c, seed=seed)
elif initializer == 'he':
    initializer = tf.keras.initializers.HeNormal(seed=seed)
else:
    raise NotImplementedError

def rhs(xy):
    return (1 - np.pi**2 - (6*np.pi)**2) * tf.sin(np.pi * xy[:, 0:1]) * tf.sin(6 * np.pi * xy[:, 1:2])

def u(xy):
    return tf.sin(np.pi * xy[:, 0:1]) * tf.sin(6 * np.pi * xy[:, 1:2])    

n_samples = nc

tx_samples = np.random.uniform(-1, 1, size=(n_samples, 2))

tx_bnd_bottom = np.concatenate([np.linspace(-1, 1, n_samples//4)[:, None], -np.ones(n_samples//4)[:, None]], axis=1)
tx_bnd_top = np.concatenate([np.linspace(-1, 1, n_samples//4)[:, None], np.ones(n_samples//4)[:, None]], axis=1)
tx_bnd_left = np.concatenate([-np.ones(n_samples//4)[:, None], np.linspace(-1, 1, n_samples//4)[:, None]], axis=1)
tx_bnd_right = np.concatenate([np.ones(n_samples//4)[:, None], np.linspace(-1, 1, n_samples//4)[:, None]], axis=1)
tx_bnd = np.concatenate([tx_bnd_bottom, tx_bnd_top, tx_bnd_left, tx_bnd_right], axis=0)

tx_samples = tf.convert_to_tensor(tx_samples, dtype=tf.float32)
tx_bnd = tf.convert_to_tensor(tx_bnd, dtype=tf.float32)
u_samples = u(tx_samples)
u_bnd = u(tx_bnd)
inputs = [tx_samples, tx_bnd]
outputs = [u_samples, rhs(tx_samples), u_bnd]

mean = tf.reduce_mean(tx_samples, axis=0)
std = tf.math.reduce_std(tx_samples, axis=0)

if initializer_st == 'sine':
    name = f'helmholtz_backbone' + '_width_' + str(width) + '_depth_' + str(depth) + \
        f'_activation_{activation_st}' + '_initializer_' + f'SineInit{init_c}' + '_epochs_' + str(epochs) + \
            '_n_' + str(nc) + '_seed_' + str(seed)
else:
    name = f'helmholtz_backbone' + '_width_' + str(width) + '_depth_' + str(depth) + \
        f'_activation_{activation_st}' + '_initializer_' + initializer_st + '_epochs_' + str(epochs) + \
            '_n_' + str(nc) + '_seed_' + str(seed)
print(name)
backbone = create_dense_model([tf.keras.layers.Normalization(mean=mean, variance=std**2)] + [width]*depth, activation=activation,
                              initializer=initializer, n_inputs=2, n_outputs=1)
model = Helmholtz2DPinn(backbone)
scheduler = tf.keras.optimizers.schedules.ExponentialDecay(1e-3, 1000, 0.95)
optimizer = tf.keras.optimizers.Adam(scheduler)
model.compile(optimizer=optimizer)

hist = model.fit_custom(inputs, outputs, epochs=epochs, print_every=1000)

backbone.save('./new_experiments/models/' + name)
save_history_csv(hist, name, './new_experiments/history')
    

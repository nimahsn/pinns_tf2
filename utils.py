from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
import tensorflow as tf


def simulate_burgers(n_samples, boundary_samples = None, random_seed = 42, dtype=tf.float32):
    """
    Simulate the burgers equation

    Args:
        n_samples (int): number of samples to generate
        boundary_samples (int, optional): number of boundary samples to generate. If None, then boundary_samples = n_samples. Defaults to None.
        random_seed (int, optional): Random seed for reproducibility. Defaults to 42.
        dtype (tf.dtype, optional): Data type of the samples. Defaults to tf.float32.

    returns:
        tf.Tensor: Samples of the burgers equation. If training = True, returns a tuple of tensors (equation_samples, initial_samples, boundary_samples).
    """
    if boundary_samples is None:
        boundary_samples = n_samples

    r = np.random.RandomState(random_seed)
    tx_samples = r.uniform(0, 1, (n_samples, 2))
    tx_samples[:, 1] = tx_samples[:, 1]*2 - 1
    tx_samples = tf.convert_to_tensor(tx_samples, dtype=dtype)
    
    y_samples = tf.zeros((n_samples, 1), dtype = dtype)

    
    tx_init = np.zeros((boundary_samples, 1))
    tx_init = np.append(tx_init, r.uniform(-1, 1, (boundary_samples, 1)), axis=1)

    tx_boundary = r.uniform(0, 1, (boundary_samples, 1))
    ones = np.ones((boundary_samples//2, 1))
    ones = np.append(ones, -np.ones((boundary_samples - boundary_samples//2, 1)), axis=0)
    tx_boundary = np.append(tx_boundary, ones, axis=1)
    r.shuffle(tx_boundary)

    tx_init = tf.convert_to_tensor(tx_init, dtype = dtype)
    tx_boundary = tf.convert_to_tensor(tx_boundary, dtype = dtype)

    y_init = tf.reshape(-tf.sin(np.pi*tx_init[..., 1]), shape = [-1, 1])
    y_boundary = tf.zeros((boundary_samples, 1), dtype = dtype)
    
    return (tx_samples, y_samples), (tx_init, y_init), (tx_boundary, y_boundary)

def plot_burgers_model(model, save_path = None):
    """
    Plot the model predictions for the Burgers equation.

    Args:
        model: A trained BurgersPinn model.
        save_path: The path to save the plot to.
    """

    num_test_samples = 1000
    t_flat = np.linspace(0, 1, num_test_samples)
    x_flat = np.linspace(-1, 1, num_test_samples)
    t, x = np.meshgrid(t_flat, x_flat)
    tx = np.stack([t.flatten(), x.flatten()], axis=-1)
    u = model.predict(tx, batch_size=num_test_samples)
    u = u.reshape(t.shape)

    # plot u(t,x) distribution as a color-map
    fig = plt.figure(figsize=(7,4))
    gs = GridSpec(2, 5)
    plt.subplot(gs[0, :])
    plt.pcolormesh(t, x, u)
    plt.xlabel('t')
    plt.ylabel('x')
    cbar = plt.colorbar(pad=0.05, aspect=10)
    cbar.set_label('u(t,x)')
    cbar.mappable.set_clim(-1, 1)
    # plot u(t=const, x) cross-sections
    t_cross_sections = [0,0.25, 0.5,0.75,1]
    for i, t_cs in enumerate(t_cross_sections):
        plt.subplot(gs[1, i])
        tx = np.stack([np.full(t_flat.shape, t_cs), x_flat], axis=-1)
        u = model.predict(tx, batch_size=num_test_samples)
        plt.plot(x_flat, u)
        plt.title('t={}'.format(t_cs))
        plt.xlabel('x')
        plt.ylabel('u(t,x)')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()
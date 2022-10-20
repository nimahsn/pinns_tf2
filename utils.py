from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
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


def simulate_wave(n_samples, dimension, phi_function, psi_function, boundary_function_start, boundary_function_end, length = 1, time = 1, random_seed = 42, dtype=tf.float32):
    """
    Simulate the wave equation in 1D or 2D with a given initial condition and Dirichlet boundary conditions.
    Args:
        n_samples (int): number of samples to generate
        dimension (int): dimension of the wave equation. Either 1 or 2.
        phi_function (function): Function that returns the initial condition of the wave equation on u.
        psi_function (function): Function that returns the initial condition of the wave equation on u_t.
        boundary_function_start (function): Function that returns the boundary condition of the wave equation on u at the start of the domain.
        boundary_function_end (function): Function that returns the boundary condition of the wave equation on u at the end of the domain.
        length (float, optional): Length of the domain. Defaults to 1.
        time (float, optional): Time frame of the simulation. Defaults to 1.
        random_seed (int, optional): Random seed for reproducibility. Defaults to 42.
        dtype (tf.dtype, optional): Data type of the samples. Defaults to tf.float32.
    
    """
    
    r = np.random.RandomState(random_seed)
    t = r.uniform(0, time, (n_samples, 1))
    x = r.uniform(0, length, (n_samples, dimension))
    tx_eqn = np.concatenate((t, x), axis = 1)

    t_init = np.zeros((n_samples, 1))
    x_init = r.uniform(0, length, (n_samples, dimension))
    tx_init = np.concatenate((t_init, x_init), axis = 1)

    t_boundary = r.uniform(0, time, (n_samples, 1))
    x_boundary = np.ones((n_samples//2, 1))*length
    x_boundary = np.append(x_boundary, np.zeros((n_samples - n_samples//2, 1)), axis=0)
    tx_boundary = np.concatenate((t_boundary, x_boundary), axis = 1)

    tx_eqn = tf.convert_to_tensor(tx_eqn, dtype = dtype)
    tx_init = tf.convert_to_tensor(tx_init, dtype = dtype)
    tx_boundary = tf.convert_to_tensor(tx_boundary, dtype = dtype)

    y_eqn = np.zeros((n_samples, 1))
    y_phi = phi_function(tx_init)
    y_psi = psi_function(tx_eqn)
    y_boundary = boundary_function_start(t_boundary[:n_samples//2])
    y_boundary = np.append(y_boundary, boundary_function_end(t_boundary[n_samples//2:]), axis=0)

    y_eqn = tf.convert_to_tensor(y_eqn, dtype = dtype)
    y_phi = tf.convert_to_tensor(y_phi, dtype = dtype)
    y_psi = tf.convert_to_tensor(y_psi, dtype = dtype)
    y_boundary = tf.convert_to_tensor(y_boundary, dtype = dtype)

    return (tx_eqn, y_eqn), (tx_init, y_phi, y_psi), (tx_boundary, y_boundary)
    

def plot_wave_model(model, length, time, save_path = None):
    """
    todo
    """
    
    def phi(tx, c=1, k=2, sd=0.5):
        t = tx[..., 0, None]
        x = tx[..., 1, None]
        z = k*x - (c*k)*t
        return tf.sin(z) * tf.exp(-(0.5*z/sd)**2)
    # du0/dt
    def psi(tx):
        with tf.GradientTape() as g:
            g.watch(tx)
            u = phi(tx)
        du_dt = g.batch_jacobian(u, tx)[..., 0]
        return du_dt

    t, x = np.meshgrid(np.linspace(0, time, 100), np.linspace(0, length, 100))
    tx = np.stack([t.flatten(), x.flatten()], axis=-1)
    u = model.predict(tx, batch_size=1000)
    print(u.shape)
    # u = u.reshape(t.shape)

    # fig = go.Figure(data=[go.Surface(x = t, y = x, z=u.reshape(t.shape))])
    # fig.update_traces(contours_z=dict(show=True, usecolormap=True,
    #                                 highlightcolor="limegreen", project_z=True))
    # fig.update_layout(title='Mt Bruno Elevation', autosize=False,
    #                 scene_camera_eye=dict(x=1.87, y=0.88, z=-0.64),
    #                 width=500, height=500,
    #                 margin=dict(l=65, r=50, b=65, t=90)
    # )

    # fig.show()

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(t.flatten(), x.flatten(), u, cmap='viridis')
    plt.show()


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
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
import plotly.graph_objects as go


def simulate_burgers(n_samples, boundary_samples = None, random_seed = 42, dtype=tf.float32) -> tuple[tuple[tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor]]:
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


def simulate_wave(n_samples, dimension, phi_function, psi_function, boundary_function_start, boundary_function_end, x_start = 0, length = 1, time = 1, random_seed = 42, dtype=tf.float32) -> tuple[tuple[tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor]]:
    """
    Simulate the wave equation in 1D or 2D with a given initial condition and Dirichlet boundary conditions.
    Args:
        n_samples (int): number of samples to generate
        dimension (int): dimension of the wave equation. Either 1 or 2.
        phi_function (function): Function that returns the initial condition of the wave equation on u.
        psi_function (function): Function that returns the initial condition of the wave equation on u_t.
        boundary_function_start (function): Function that returns the boundary condition of the wave equation on u at the start of the domain.
        boundary_function_end (function): Function that returns the boundary condition of the wave equation on u at the end of the domain.
        x_start (float, optional): Start of the domain. Defaults to 0.
        length (float, optional): Length of the domain. Defaults to 1.
        time (float, optional): Time frame of the simulation. Defaults to 1.
        random_seed (int, optional): Random seed for reproducibility. Defaults to 42.
        dtype (tf.dtype, optional): Data type of the samples. Defaults to tf.float32.

    Returns:
        tuple[tuple[tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor]]: Samples of the wave equation. Returns a tuple of tensors (equation_samples, initial_samples, boundary_samples).
    
    """
    
    r = np.random.RandomState(random_seed)
    t = r.uniform(0, time, (n_samples, 1))
    x = r.uniform(x_start, x_start + length, (n_samples, dimension))
    tx_eqn = np.concatenate((t, x), axis = 1)

    t_init = np.zeros((n_samples, 1))
    x_init = r.uniform(x_start, x_start + length, (n_samples, dimension))
    tx_init = np.concatenate((t_init, x_init), axis = 1)

    t_boundary = r.uniform(0, time, (n_samples, 1))
    x_boundary = np.ones((n_samples//2, 1))*(x_start + length)
    x_boundary = np.append(x_boundary, np.ones((n_samples - n_samples//2, 1)) * x_start, axis=0)
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


def simulate_heat(n_samples, phi_function, boundary_function, length, time, random_seed = 42, dtype=tf.float32) -> tuple[tuple[tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor]]:
    """
    Simulate the heat equation in 1D with a given initial condition and Dirichlet boundary conditions.
    Args:
        n_samples (int): number of samples to generate
        phi_function (function): Function that returns the initial condition of the heat equation on u.
        boundary_function (function): Function that returns the boundary condition of the heat equation on u.
        length (float): Length of the domain.
        time (float): Time frame of the simulation.
        random_seed (int, optional): Random seed for reproducibility. Defaults to 42.
        dtype (tf.dtype, optional): Data type of the samples. Defaults to tf.float32.

    Returns:
        tuple[tuple[tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor]]: Samples of the heat equation. Returns a tuple of tensors (equation_samples, initial_samples, boundary_samples).
    """

    r = np.random.RandomState(random_seed)
    t = r.uniform(0, time, (n_samples, 1))
    x = r.uniform(0, length, (n_samples, 1))
    tx_eqn = np.concatenate((t, x), axis = 1)

    t_init = np.zeros((n_samples, 1))
    x_init = r.uniform(0, length, (n_samples, 1))
    tx_init = np.concatenate((t_init, x_init), axis = 1)

    t_boundary = r.uniform(0, time, (n_samples, 1))
    x_boundary = np.ones((n_samples//2, 1))*length
    x_boundary = np.append(x_boundary, np.zeros((n_samples - n_samples//2, 1)), axis=0)
    tx_boundary = np.concatenate((t_boundary, x_boundary), axis = 1)

    tx_eqn = tf.convert_to_tensor(tx_eqn, dtype = dtype)
    tx_init = tf.convert_to_tensor(tx_init, dtype = dtype)
    tx_boundary = tf.convert_to_tensor(tx_boundary, dtype = dtype)

    y_eqn = tf.zeros((n_samples, 1))
    y_phi = phi_function(tx_init)
    y_boundary = boundary_function(tx_boundary)

    # y_eqn = tf.convert_to_tensor(y_eqn, dtype = dtype)
    # y_phi = tf.convert_to_tensor(y_phi, dtype = dtype)
    # y_boundary = tf.convert_to_tensor(y_boundary, dtype = dtype)

    return (tx_eqn, y_eqn), (tx_init, y_phi), (tx_boundary, y_boundary)


def simulate_schrodinger(n_samples, init_function, x_start, length, time, random_seed = 42, dtype = tf.float32):

    r = np.random.RandomState(random_seed)
    
    t = r.uniform(0, time, (n_samples, 1))
    x = r.uniform(x_start, x_start + length, (n_samples, 1))
    tx_eqn = np.concatenate((t, x), axis = 1)

    t_init = np.zeros((n_samples, 1))
    x_init = r.uniform(x_start, x_start + length, (n_samples, 1))
    tx_init = np.concatenate((t_init, x_init), axis = 1)

    t_boundary = r.uniform(0, time, (n_samples, 1))
    x_boundary = np.ones((n_samples//2, 1))*x_start
    x_boundary = np.append(x_boundary, np.ones((n_samples - n_samples//2, 1)) * (x_start + length), axis=0)
    tx_boundary = np.concatenate((t_boundary, x_boundary), axis = 1)

    tx_eqn = tf.convert_to_tensor(tx_eqn, dtype = dtype)
    tx_init = tf.convert_to_tensor(tx_init, dtype = dtype)
    tx_boundary = tf.convert_to_tensor(tx_boundary, dtype = dtype)

    y_eqn = tf.zeros((n_samples, 2), dtype = dtype)
    y_init = init_function(tx_init)

    return (tx_eqn, y_eqn), (tx_init, y_init), tx_boundary



def plot_wave_model(model, x_start, length, time, interactive = False, save_path = None) -> None:
    """
    Plot the solution of the wave equation for a given model.
    Args:
        model (tf.keras.Model): Model that predicts the solution of the wave equation.
        x_start (float): Start of the domain.
        length (float): Length of the domain.
        time (float): Time frame of the simulation.
        interactive (bool, optional): If True, the plot is interactive. Defaults to False.
        save_path (str, optional): Path to save the plot. Defaults to None.
    """

    t, x = np.meshgrid(np.linspace(0, time, 100), np.linspace(x_start, x_start + length, 100))  
    tx = np.stack([t.flatten(), x.flatten()], axis=-1)
    u = model.predict(tx, batch_size=1000)

    if interactive:
        fig = go.Figure(data=[go.Surface(x = t, y = x, z=u.reshape(t.shape))])
        fig.update_traces(contours_z=dict(show=True, usecolormap=True,
                                        highlightcolor="limegreen", project_z=True))
        fig.update_layout(title='Mave PINN', autosize=True, scene=dict(
                            xaxis_title='t',
                            yaxis_title='x',
                            zaxis_title='y',
                        ),
                        scene_camera_eye=dict(x=1.87, y=0.88, z=-0.64),
                        width=500, height=500,
                        margin=dict(l=65, r=50, b=65, t=90)
        )
        fig.show()

    else:
        fig = plt.figure(figsize=(30, 90))
        ax = fig.add_subplot(311, projection='3d')
        surf = ax.scatter(t, x, np.reshape(u, t.shape), cmap='viridis', alpha=0.6)
        ax.set_xlabel('t')
        ax.set_ylabel('x')
        ax.set_zlabel('u')
        ax.azim = 5
        ax.elev = 20
        # fig.colorbar(surf)


        ax = fig.add_subplot(312, projection='3d')
        surf = ax.scatter(t, x, np.reshape(u, t.shape), cmap='viridis', alpha=0.6)
        ax.set_xlabel('t')
        ax.set_ylabel('x')
        ax.set_zlabel('u')
        ax.azim = 45
        ax.elev = 20
        # fig.colorbar(surf)


        ax = fig.add_subplot(313, projection='3d')
        surf = ax.scatter(t, x, np.reshape(u, t.shape), cmap='viridis', alpha=0.6)
        ax.azim = 85
        ax.elev = 20
        ax.set_xlabel('t')
        ax.set_ylabel('x')
        ax.set_zlabel('u')
        # fig.colorbar(surf)


        if save_path:
            plt.savefig(save_path)
        plt.show()


def plot_burgers_model(model, save_path = None) -> None:
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


def plot_heat_model(model, length, time, save_path = None) -> None:
    """
    Plot the model predictions for the heat equation.
    Args:
        model: A trained HeatPinn model.
        length: The length of the domain.
        time: The time frame of the simulation.
        save_path: The path to save the plot to.
    """
    num_test_samples = 1000
    t_flat = np.linspace(0, time, num_test_samples)
    x_flat = np.linspace(0, length, num_test_samples)
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
    t_cross_sections = [0, time/4, time/2, 3*time/4, time]
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


def plot_schrodinger_model(model, x_start, length, time, save_path = None) -> None:
    """
    Plot the model predictions for the Schrodinger equation.
    Args:
        model: A trained SchrodingerPinn model.
        length: The length of the domain.
        save_path: The path to save the plot to.
    """
    num_test_samples = 1000
    t_flat = np.linspace(0, time, num_test_samples)
    x_flat = np.linspace(x_start, x_start + length, num_test_samples)
    t, x = np.meshgrid(t_flat, x_flat)
    tx = np.stack([t.flatten(), x.flatten()], axis=-1)
    h = model.predict(tx, batch_size=num_test_samples)
    u = tf.abs(tf.complex(h[:,0:1], h[:,1:2]))
    u = tf.reshape(u, t.shape)

    # plot u(t,x) distribution as a color-map
    fig = plt.figure(figsize=(7,4))
    gs = GridSpec(2, 5)
    plt.subplot(gs[0, :])
    plt.pcolormesh(t, x, u)
    plt.xlabel('t')
    plt.ylabel('x')
    cbar = plt.colorbar(pad=0.05, aspect=10)
    cbar.set_label('|h(t,x)|')
    cbar.mappable.set_clim(-1, 1)
    # plot u(t=const, x) cross-sections
    t_cross_sections = [0, time/4, time/2, 3*time/4, time]
    for i, t_cs in enumerate(t_cross_sections):
        plt.subplot(gs[1, i])
        tx = np.stack([np.full(t_flat.shape, t_cs), x_flat], axis=-1)
        h = model.predict(tx, batch_size=num_test_samples)
        u = tf.abs(tf.complex(h[:,0:1], h[:,1:2]))
        plt.plot(x_flat, u)
        plt.title('t={}'.format(t_cs))
        plt.xlabel('x')
        plt.ylabel('|h(t,x)|')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()
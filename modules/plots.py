"""
Utility module for plotting models and losses.
"""

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import plotly.graph_objects as go
import tensorflow as tf
from modules.models import LOSS_RESIDUAL, LOSS_BOUNDARY, LOSS_INITIAL, MEAN_ABSOLUTE_ERROR


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


def plot_poisson_model(model, x_start, length, save_path = None) -> None:
    """
    Plot the model predictions for the Poisson equation.
    Args:
        model: A trained PoissonPinn model.
        length: The length of the domain.
        save_path: The path to save the plot to.
    """
    num_test_samples = 1000
    x = np.linspace(x_start, x_start + length, num_test_samples)[:, np.newaxis]
    u = model.predict(x, batch_size=num_test_samples)

    # plot u(x) distribution as a color-map
    fig, ax = plt.subplots(figsize = (7,4))
    ax.plot(x.flatten(), u.flatten())
    ax.set_xlabel('x')
    ax.set_ylabel('u(x)')
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


def plot_advection_model(model, x_start = 0.0, length = 1.0, save_path = None) -> None:
    """
    Plot the model predictions for the advection equation.
    Args:
        model: A trained AdvectionPinn model.
        length: The length of the domain.
        save_path: The path to save the plot to.
    """
    num_test_samples = 1000
    x = np.linspace(x_start, x_start + length, num_test_samples)[:, np.newaxis]
    u = model.predict(x, batch_size=num_test_samples)

    # plot u(x) distribution as a color-map
    fig, ax = plt.subplots(figsize = (7,4))
    ax.plot(x.flatten(), u.flatten())
    ax.set_xlabel('x')
    ax.set_ylabel('u(x)')
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


def plot_training_loss(history, x_scale = "linear", y_scale = "linear", save_path=None):
    """
    Plot the training residual, initial, and boundary losses separately.
    Args:
        history: The history object returned by the model.fit() method.
        x_scale: The scale of the x-axis.
        y_scale: The scale of the y-axis.
    """
    plt.figure(figsize=(10, 5), dpi = 150)
    plt.xscale(x_scale)
    plt.yscale(y_scale)
    if len(history[LOSS_INITIAL]) > 0:
        plt.plot(history[LOSS_INITIAL], label='initial loss')
    if len(history[LOSS_BOUNDARY]) > 0:
        plt.plot(history[LOSS_BOUNDARY], label='boundary loss')
    if len(history[LOSS_RESIDUAL]) > 0:
        plt.plot(history[LOSS_RESIDUAL], label='residual loss')
    if len(history[MEAN_ABSOLUTE_ERROR]) > 0:
        plt.plot(history[MEAN_ABSOLUTE_ERROR], label='mean absolute error')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


def plot_pointwise_error(y_true, y_pred, x, figsize = (10, 5), save_path=None):
    """
    Plot the pointwise error between the true and predicted values.
    Args:
        y_true: The true values.
        y_pred: The predicted values.
        x: The x-values.
    """
    plt.figure(figsize=figsize, dpi = 150)
    plt.plot(x, np.abs(y_true - y_pred))
    plt.xlabel('x')
    plt.ylabel('Absolute error')
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()
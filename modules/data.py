"""
Module for the data generation of the heat, wave, schrodinger, burgers, and poisson equations.
"""

import numpy as np
import tensorflow as tf
from typing import Callable, Tuple


def simulate_burgers(n_samples, init_function = None, boundary_function = None, random_seed = 42, dtype=tf.float32) -> Tuple[Tuple[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]:
    """
    Simulate the burgers equation

    Args:
        n_samples (int): number of samples to generate
        init_function (function, optional): Function that returns the initial condition of the burgers equation. If None, sin(pi*x) is used. Defaults to None.
        boundary_function (function, optional): Function that returns the boundary condition of the burgers equation. If None, 0 is used. Defaults to None.
        boundary_samples (int, optional): number of boundary samples to generate. If None, then boundary_samples = n_samples. Defaults to None.
        random_seed (int, optional): Random seed for reproducibility. Defaults to 42.
        dtype (tf.dtype, optional): Data type of the samples. Defaults to tf.float32.

    returns:
        tf.Tensor: Samples of the burgers equation. If training = True, returns a tuple of tensors (equation_samples, initial_samples, n_samples).
    """

    if init_function is None:
        def init_function(tx):
            return -tf.sin(np.pi*tx[:, 1:])

    if boundary_function is None:
        def boundary_function(tx):
            return tf.zeros_like(tx[:, 1:])

    r = np.random.RandomState(random_seed)
    tx_samples = r.uniform(0, 1, (n_samples, 2))
    tx_samples[:, 1] = tx_samples[:, 1]*2 - 1
    tx_samples = tf.convert_to_tensor(tx_samples, dtype=dtype)
    
    y_samples = tf.zeros((n_samples, 1), dtype = dtype)

    
    tx_init = np.zeros((n_samples, 1))
    tx_init = np.append(tx_init, r.uniform(-1, 1, (n_samples, 1)), axis=1)

    tx_boundary = r.uniform(0, 1, (n_samples, 1))
    ones = np.ones((n_samples//2, 1))
    ones = np.append(ones, -np.ones((n_samples - n_samples//2, 1)), axis=0)
    tx_boundary = np.append(tx_boundary, ones, axis=1)
    r.shuffle(tx_boundary)

    tx_init = tf.convert_to_tensor(tx_init, dtype = dtype)
    tx_boundary = tf.convert_to_tensor(tx_boundary, dtype = dtype)

    y_init = init_function(tx_init)
    y_boundary = boundary_function(tx_boundary)
    
    return (tx_samples, y_samples), (tx_init, y_init), (tx_boundary, y_boundary)


def simulate_wave(n_samples, phi_function, psi_function, boundary_function, x_start = 0, length = 1, time = 1, random_seed = 42, dtype = tf.float32) -> Tuple[Tuple[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]:
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
    x = r.uniform(x_start, x_start + length, (n_samples, 1))
    tx_eqn = np.concatenate((t, x), axis = 1)

    t_init = np.zeros((n_samples, 1))
    x_init = r.uniform(x_start, x_start + length, (n_samples, 1))
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
    y_psi = psi_function(tx_init)
    y_boundary = boundary_function(tx_boundary)

    return (tx_eqn, y_eqn), (tx_init, y_phi, y_psi), (tx_boundary, y_boundary)


def simulate_heat(n_samples, phi_function, boundary_function, length, time, random_seed = 42, dtype=tf.float32) -> Tuple[Tuple[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]:
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


def simulate_poisson(n_samples, rhs_function, boundary_function, x_start: float = 0.0, length: float = 1.0, random_seed = 42, dtype = tf.float32) -> Tuple[Tuple[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]:
    """
    Simulate the Poisson equation in 1D with a given right hand side and Dirichlet boundary conditions.
    Args:
        n_samples (int): number of samples to generate
        rhs_function (function): Function that returns the right hand side of the Poisson equation.
        boundary_function (function): Function that returns the boundary condition of the Poisson equation on u.
        boundary_start (float, optional): Start of the boundary. Defaults to 0.0.
        length (float, optional): Length of the domain. Defaults to 1.0.
        random_seed (int, optional): Random seed for reproducibility. Defaults to 42.
        dtype (tf.dtype, optional): Data type of the samples. Defaults to tf.float32.
    """

    r = np.random.RandomState(random_seed)
    
    x_eqn = r.uniform(x_start, x_start + length, (n_samples, 1))

    x_boundary = np.ones((n_samples//2, 1)) * x_start
    x_boundary = np.append(x_boundary, np.ones((n_samples - n_samples//2, 1)) * (x_start + length), axis=0)

    x_eqn = tf.convert_to_tensor(x_eqn, dtype = dtype)
    x_boundary = tf.convert_to_tensor(x_boundary, dtype = dtype)

    rhs_eqn = rhs_function(x_eqn)
    u_boundary = boundary_function(x_boundary)

    return (x_eqn, rhs_eqn), (x_boundary, u_boundary)


def simulate_advection(n_samples, boundary_function: Callable = None, x_start: float = 0.0, length: float = 1, random_seed = 42, dtype = tf.float32):
    """
    Simulate the steady advection diffusion equation in 1D with a given boundary conditions.
    Args:
        n_samples (int): number of samples to generate
        boundary_function (function): Function that returns the boundary condition of the advection diffusion equation on u.\
            If None, the boundary condition is set to zero on start and one on end. Defaults to None.
        x_start (float, optional): Start of the boundary. Defaults to 0.0.
        length (float, optional): Length of the domain. Defaults to 1.0.
        random_seed (int, optional): Random seed for reproducibility. Defaults to 42.
        dtype (tf.dtype, optional): Data type of the samples. Defaults to tf.float32.
    """

    if boundary_function is None:
        def boundary_function(x):
            return tf.where(x == x_start, 0.0, 1.0)

    r = np.random.RandomState(random_seed)
    
    x_eqn = r.uniform(x_start, x_start + length, (n_samples, 1))

    x_boundary = np.ones((n_samples//2, 1)) * x_start
    x_boundary = np.append(x_boundary, np.ones((n_samples - n_samples//2, 1)) * (x_start + length), axis=0)

    x_eqn = tf.convert_to_tensor(x_eqn, dtype = dtype)
    x_boundary = tf.convert_to_tensor(x_boundary, dtype = dtype)

    f_eqn = tf.zeros((n_samples, 1))
    u_boundary = boundary_function(x_boundary)

    return (x_eqn, f_eqn), (x_boundary, u_boundary)


def simulate_schrodinger(n_samples, init_function, x_start, length, time, random_seed = 42, dtype = tf.float32):
    """
    Simulate the Schrodinger equation in 1D with a given initial condition.
    Args:
        n_samples (int): number of samples to generate
        init_function (function): Function that returns the initial condition of the Schrodinger equation.
        x_start (float): Start of the boundary.
        length (float): Length of the domain.
        time (float): Time of the simulation.
        random_seed (int, optional): Random seed for reproducibility. Defaults to 42.
        dtype (tf.dtype, optional): Data type of the samples. Defaults to tf.float32.

    Returns:
        Tuple[Tuple[tf.tensor, tf.tensor], Tuple[tf.tensor, tf.tensor], tf.tensor]: Tuple of tuples of tensors. \
            The first tuple contains the equation samples, the second tuple the initial condition samples and the third tensor the boundary condition samples. \
    """

    r = np.random.RandomState(random_seed)
    
    t = r.uniform(0, time, (n_samples, 1))
    x = r.uniform(x_start, x_start + length, (n_samples, 1))
    tx_eqn = np.concatenate((t, x), axis = 1)

    t_init = np.zeros((n_samples, 1))
    x_init = r.uniform(x_start, x_start + length, (n_samples, 1))
    tx_init = np.concatenate((t_init, x_init), axis = 1)

    t_boundary = r.uniform(0, time, (n_samples, 1))
    x_boundary_start = np.ones((n_samples, 1)) * x_start
    x_boundary_end = np.ones((n_samples, 1)) * (x_start + length)
    txx_boundary = np.concatenate([t_boundary, x_boundary_start, x_boundary_end], axis = 1)

    tx_eqn = tf.convert_to_tensor(tx_eqn, dtype = dtype)
    tx_init = tf.convert_to_tensor(tx_init, dtype = dtype)
    txx_boundary = tf.convert_to_tensor(txx_boundary, dtype = dtype)

    y_eqn = tf.zeros((n_samples, 2), dtype = dtype)
    y_init = init_function(tx_init)

    return (tx_eqn, y_eqn), (tx_init, y_init), txx_boundary
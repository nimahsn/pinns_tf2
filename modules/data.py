"""
Module for the data generation of the heat, wave, schrodinger, burgers, and poisson equations.
"""
import tensorflow as tf
from typing import Callable, Tuple
import numpy as np


def simulate_burgers(n_samples, init_function=None, boundary_function=None, n_init=None, n_bndry=None, random_seed=42, dtype=tf.float32) \
    -> Tuple[Tuple[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]:
    """
    Simulate the burgers equation

    Args:
        n_samples (int): number of samples to generate
        init_function (function, optional): Function that returns the initial condition of the burgers equation. If None, sin(pi*x) is used. Defaults to None.
        boundary_function (function, optional): Function that returns the boundary condition of the burgers equation. If None, 0 is used. Defaults to None.
        boundary_samples (int, optional): number of boundary samples to generate. If None, then boundary_samples = n_samples. Defaults to None.
        n_init (int, optional): number of initial samples to generate. If None, then n_init = n_samples. Defaults to None.
        n_bndry (int, optional): number of boundary samples to generate. If None, then n_bndry = n_samples. Defaults to None.
        random_seed (int, optional): Random seed for reproducibility. Defaults to 42.
        dtype (tf.dtype, optional): Data type of the samples. Defaults to tf.float32.

    returns:
        tf.Tensor: Samples of the burgers equation. If training = True, returns a tuple of tensors (equation_samples, initial_samples, n_samples).
    """

    if n_init is None:
        n_init = n_samples
    if n_bndry is None:
        n_bndry = n_samples

    if init_function is None:
        def init_function(tx):
            return -tf.sin(np.pi*tx[:, 1:])

    if boundary_function is None:
        def boundary_function(tx):
            return tf.zeros_like(tx[:, 1:])

    assert n_bndry % 2 == 0, "n_bndry must be even"

    tx_samples = tf.random.uni.uniform((n_samples, 2), -1, 1, dtype=dtype, seed=random_seed)
    tx_samples[:, 1] = tx_samples[:, 1] * 2 - 1
    y_samples = tf.zeros((n_samples, 1), dtype=dtype)

    tx_init = tf.zeros((n_init, 1), dtype=dtype)
    tx_init = tf.concat(tx_init, tf.random.uniform((n_init, 1), -1, 1, seed=random_seed, dtype=dtype), axis=1)
    y_init = init_function(tx_init)

    tx_boundary = tf.random.uniform((n_bndry, 1), 0, 1, dtype=dtype, seed=random_seed)
    ones = tf.ones((n_bndry//2, 1), dtype=dtype)
    ones = tf.concat([ones, -tf.ones((n_bndry//2, 1), dtype=dtype)], axis=0)
    tx_boundary = tf.concat([tx_boundary, ones], axis=1)
    tx_boundary = tf.random.shuffle(tx_boundary, seed=random_seed)
    y_boundary = boundary_function(tx_boundary)
    
    return (tx_samples, y_samples), (tx_init, y_init), (tx_boundary, y_boundary)


def simulate_wave(n_samples, phi_function, psi_function, boundary_function, x_start=0.0, length=1.0, time=1.0, n_init=None, n_bndry=None, \
    random_seed=42, dtype = tf.float32) -> Tuple[Tuple[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]:
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
        n_init (int, optional): number of initial samples to generate. If None, then n_init = n_samples. Defaults to None.
        n_bndry (int, optional): number of boundary samples to generate. If None, then n_bndry = n_samples. Defaults to None.
        random_seed (int, optional): Random seed for reproducibility. Defaults to 42.
        dtype (tf.dtype, optional): Data type of the samples. Defaults to tf.float32.

    Returns:
        tuple[tuple[tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor]]: Samples of the wave equation. Returns a tuple of tensors (equation_samples, initial_samples, boundary_samples).
    
    """
    if n_init is None:
        n_init = n_samples
    if n_bndry is None:
        n_bndry = n_samples
    
    assert n_bndry % 2 == 0, "n_bndry must be even"
    
    t = tf.random.uniform((n_samples, 1), 0, time, dtype=dtype, seed=random_seed)
    x = tf.random.uniform((n_samples, 1), x_start, x_start + length, dtype=dtype, seed=random_seed)
    tx_eqn = tf.concat((t, x), axis=1)
    y_eqn = tf.zeros((n_samples, 1), dtype=dtype)

    t_init = tf.zeros((n_init, 1))
    x_init = tf.random.uniform((n_init, 1), x_start, x_start + length, dtype=dtype, seed=random_seed)
    tx_init = tf.concat((t_init, x_init), axis=1)

    t_boundary = tf.random.uniform((n_bndry, 1), 0, time, dtype=dtype, seed=random_seed)
    x_boundary = tf.ones((n_bndry//2, 1), dtype=dtype) * (x_start + length)
    x_boundary = tf.concat([x_boundary, tf.ones((n_bndry//2, 1), dtype=dtype) * x_start], axis=0)
    x_boundary = tf.random.shuffle(x_boundary, seed=random_seed)
    tx_boundary = tf.concat([t_boundary, x_boundary], axis=1)

    y_phi = phi_function(tx_init)
    y_psi = psi_function(tx_init)
    y_boundary = boundary_function(tx_boundary)

    return (tx_eqn, y_eqn), (tx_init, y_phi, y_psi), (tx_boundary, y_boundary)


def simulate_heat(n_samples, phi_function, boundary_function, length, time, n_init=None, n_bndry=None, random_seed=2, dtype=tf.float32) \
    -> Tuple[Tuple[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]:
    """
    Simulate the heat equation in 1D with a given initial condition and Dirichlet boundary conditions.
    Args:
        n_samples (int): number of samples to generate
        phi_function (function): Function that returns the initial condition of the heat equation on u.
        boundary_function (function): Function that returns the boundary condition of the heat equation on u.
        length (float): Length of the domain.
        time (float): Time frame of the simulation.
        n_init (int, optional): number of initial samples to generate. If None, then n_init = n_samples. Defaults to None.
        n_bndry (int, optional): number of boundary samples to generate. If None, then n_bndry = n_samples. Defaults to None.
        random_seed (int, optional): Random seed for reproducibility. Defaults to 42.
        dtype (tf.dtype, optional): Data type of the samples. Defaults to tf.float32.

    Returns:
        tuple[tuple[tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor]]: Samples of the heat equation. Returns a tuple of tensors (equation_samples, initial_samples, boundary_samples).
    """
    if n_init is None:
        n_init = n_samples
    if n_bndry is None:
        n_bndry = n_samples
    assert n_bndry % 2 == 0, "n_bndry must be even"

    t = tf.random.uniform((n_samples, 1), 0, time, dtype=dtype, seed=random_seed)
    x = tf.random.uniform((n_samples, 1), 0, length, dtype=dtype, seed=random_seed)
    tx_eqn = tf.concat((t, x), axis=1)

    t_init = tf.zeros((n_init, 1), dtype=dtype)
    x_init = tf.random.uniform((n_init, 1), 0, length, dtype=dtype, seed=random_seed)
    tx_init = tf.concat((t_init, x_init), axis=1)

    t_boundary = tf.random.uniform((n_bndry, 1), 0, time, dtype=dtype, seed=random_seed)
    x_boundary = tf.ones((n_bndry//2, 1), dtype=dtype) * length
    x_boundary = tf.concat([x_boundary, tf.zeros((n_bndry//2, 1))], axis=0)
    x_boundary = tf.random.shuffle(x_boundary, seed=random_seed)
    tx_boundary = tf.concat([t_boundary, x_boundary], axis=1)

    y_eqn = tf.zeros((n_samples, 1), dtype=dtype)
    y_phi = phi_function(tx_init)
    y_boundary = boundary_function(tx_boundary)

    return (tx_eqn, y_eqn), (tx_init, y_phi), (tx_boundary, y_boundary)


def simulate_poisson(n_samples, rhs_function, boundary_function, x_start: float = 0.0, length: float = 1.0, n_bndry=None, random_seed=42, \
    dtype=tf.float32) -> Tuple[Tuple[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]:
    """
    Simulate the Poisson equation in 1D with a given right hand side and Dirichlet boundary conditions.
    Args:
        n_samples (int): number of samples to generate
        rhs_function (function): Function that returns the right hand side of the Poisson equation.
        boundary_function (function): Function that returns the boundary condition of the Poisson equation on u.
        boundary_start (float, optional): Start of the boundary. Defaults to 0.0.
        length (float, optional): Length of the domain. Defaults to 1.0.
        n_bndry (int, optional): number of boundary samples to generate. If None, then n_bndry = n_samples. Defaults to None.
        random_seed (int, optional): Random seed for reproducibility. Defaults to 42.
        dtype (tf.dtype, optional): Data type of the samples. Defaults to tf.float32.
    """
    if n_bndry is None:
        n_bndry = n_samples
    assert n_bndry % 2 == 0, "n_bndry must be even"
    
    x_eqn = tf.random.uniform((n_samples, 1), x_start, x_start + length, dtype=dtype, seed=random_seed)
    rhs_eqn = rhs_function(x_eqn)

    x_boundary = tf.ones((n_bndry//2, 1), dtype=dtype) * x_start
    x_boundary = tf.concat([x_boundary, tf.ones((n_bndry//2, 1), dtype=dtype) * (x_start + length)], axis=0)
    x_boundary = tf.random.shuffle(x_boundary, seed=random_seed)
    u_boundary = boundary_function(x_boundary)

    return (x_eqn, rhs_eqn), (x_boundary, u_boundary)

def simulate_advection(n_samples, boundary_function: Callable = None, x_start: float = 0.0, length: float = 1, n_bndry=None, \
     random_seed=42, dtype=tf.float32):
    """
    Simulate the steady advection diffusion equation in 1D with a given boundary conditions.
    Args:
        n_samples (int): number of samples to generate
        boundary_function (function): Function that returns the boundary condition of the advection diffusion equation on u.\
            If None, the boundary condition is set to zero on start and one on end. Defaults to None.
        x_start (float, optional): Start of the boundary. Defaults to 0.0.
        length (float, optional): Length of the domain. Defaults to 1.0.
        n_bndry (int, optional): number of boundary samples to generate. If None, then n_bndry = n_samples. Defaults to None.
        random_seed (int, optional): Random seed for reproducibility. Defaults to 42.
        dtype (tf.dtype, optional): Data type of the samples. Defaults to tf.float32.
    """
    if n_bndry is None:
        n_bndry = n_samples
    assert n_bndry % 2 == 0, "n_bndry must be even"

    if boundary_function is None:
        def boundary_function(x):
            return tf.where(x == x_start, 0.0, 1.0)
    
    x_eqn = tf.random.uniform((n_samples, 1), x_start, x_start + length, dtype=dtype, seed=random_seed)
    f_eqn = tf.zeros((n_samples, 1))

    x_boundary = tf.ones((n_bndry//2, 1), dtype=dtype) * x_start
    x_boundary = tf.concat([x_boundary, tf.ones((n_bndry//2, 1), dtype=dtype) * (x_start + length)], axis=0)
    x_boundary = tf.random.shuffle(x_boundary, seed=random_seed)
    u_boundary = boundary_function(x_boundary)

    return (x_eqn, f_eqn), (x_boundary, u_boundary)

def simulate_schrodinger(n_samples, init_function, x_start, length, time, n_init=None, n_bndry=None, random_seed=42, dtype=tf.float32):
    """
    Simulate the Schrodinger equation in 1D with a given initial condition.
    Args:
        n_samples (int): number of samples to generate
        init_function (function): Function that returns the initial condition of the Schrodinger equation.
        x_start (float): Start of the boundary.
        length (float): Length of the domain.
        time (float): Time of the simulation.
        random_seed (int, optional): Random seed for reproducibility. Defaults to 42.
        n_init (int, optional): number of initial condition samples to generate. If None, then n_init = n_samples. Defaults to None.
        n_bndry (int, optional): number of boundary samples to generate. If None, then n_bndry = n_samples. Defaults to None.
        dtype (tf.dtype, optional): Data type of the samples. Defaults to tf.float32.

    Returns:
        Tuple[Tuple[tf.tensor, tf.tensor], Tuple[tf.tensor, tf.tensor], tf.tensor]: Tuple of tuples of tensors. \
            The first tuple contains the equation samples, the second tuple the initial condition samples and the third tensor the boundary condition samples. \
    """
    if n_init is None:
        n_init = n_samples
    if n_bndry is None:
        n_bndry = n_samples

    
    t = tf.random.uniform((n_samples, 1), 0, time, dtype=dtype, seed=random_seed)
    x = tf.random.uniform((n_samples, 1), x_start, x_start + length, dtype=dtype, seed=random_seed)
    tx_eqn = tf.concat((t, x), axis=1)
    y_eqn = tf.zeros((n_samples, 2), dtype=dtype)

    t_init = tf.zeros((n_init, 1), dtype=dtype)
    x_init = tf.random.uniform((n_init, 1), x_start, x_start + length, dtype=dtype, seed=random_seed)
    tx_init = tf.concat((t_init, x_init), axis=1)
    y_init = init_function(tx_init)

    t_boundary = tf.random.uniform((n_bndry, 1), 0, time, dtype=dtype, seed=random_seed)
    x_boundary_start = tf.ones((n_bndry, 1), dtype=dtype) * x_start
    x_boundary_end = tf.ones((n_bndry, 1), dtype=dtype) * (x_start + length)
    txx_boundary = tf.concat([t_boundary, x_boundary_start, x_boundary_end], axis=1)

    return (tx_eqn, y_eqn), (tx_init, y_init), txx_boundary


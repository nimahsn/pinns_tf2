from typing import Dict
from tensorflow import keras
import tensorflow as tf
import numpy as np
import time

LOSS_RESIDUAL = "loss_residual"
LOSS_INITIAL = "loss_initial"
LOSS_BOUNDARY = "loss_boundary"


def _create_history_dict():
    return {
        LOSS_RESIDUAL: [],
        LOSS_INITIAL: [],
        LOSS_BOUNDARY: [],
    }


def _add_to_history_dict(history_dict, loss_residual, loss_initial, loss_boundary):
  if loss_residual is not None:
    history_dict[LOSS_RESIDUAL].append(loss_residual)
  if loss_initial is not None:
    history_dict[LOSS_INITIAL].append(loss_initial)
  if loss_boundary is not None:
    history_dict[LOSS_BOUNDARY].append(loss_boundary)


class BurgersPinn(keras.Model):
  def __init__(self, network, nu, n_inputs=2, n_outputs=1):
    super().__init__()
    self.network = network
    self.nu = nu

  def fit(self, inputs, labels, epochs, optimizer, progress_interval=500) -> dict[str, list[float]]:
    """
    Train the model with the given inputs and optimizer.

    Args:
      inputs: A list of tensors, where the first tensor is the equation data,
        the second tensor is the initial condition data, and the third tensor
        is the boundary condition data.
      epochs: The number of epochs to train for.
      optimizer : The optimizer to use for training.
      progress_interval: The number of epochs between each progress report.

    Returns:
        A dictionary containing the loss history for each loss function.
    """
    history_dict = _create_history_dict()
    start_time = time.time()

    for epoch in range(epochs):
      with tf.GradientTape() as tape:
        burgers_eq, u_init, u_bndry = self.call(inputs)
        loss_residual = tf.reduce_mean(burgers_eq**2) 
        loss_init = tf.reduce_mean(tf.square(u_init - labels[1]))
        loss_boundary = tf.reduce_mean(tf.square(u_bndry - labels[2]))
        loss = loss_residual + loss_init + loss_boundary
      grads = tape.gradient(loss, self.trainable_weights)
      optimizer.apply_gradients(zip(grads, self.trainable_weights))
      _add_to_history_dict(history_dict, loss_residual, loss_init, loss_boundary)
      if epoch % progress_interval == 0:
        print(f"Epoch: {epoch} Loss: {loss.numpy():.4f} Total Elapsed Time: {time.time() - start_time:.2f}")
    return history_dict

  
  @tf.function
  def input_gradient(self, x):
    """
    Compute the first and second derivatives of the network output with respect to the inputs.
    
    Args:
      x: input tensor of shape (n_inputs, 2)

    returns:
      u: network output of shape (n_inputs, 1)
      u_t: first derivative of u with respect to t
      u_x: first derivative of u with respect to x
      u_xx: second derivative of u with respect to x
    """
    with tf.GradientTape() as g2tape: # grad tape for getting second order derivatives
      g2tape.watch(x) # gradients w.r.t. inputs
      with tf.GradientTape() as gtape: # grad tape for first order drivatives
        gtape.watch(x)
        u = self.network(x)
        
      first_order = gtape.batch_jacobian(u, x)
      u_t = first_order[..., 0]
      u_x = first_order[..., 1]
      
    u_xx = g2tape.batch_jacobian(u_x, x)[..., 1]
    return u, u_t, u_x, u_xx
  
  
  def call(self, inputs):
    """
    Performs forward pass of the model, computing the PDE residual and the initial and boundary conditions.

    Args:
      inputs: A list of tensors, where the first tensor is the equation data,
        the second tensor is the initial condition data, and the third tensor
        is the boundary condition data.

    Returns:
        burgers_eq: The PDE residual of shape (n_inputs, 1)
        u_init: The initial condition residual of shape (n_inputs, 1)
        u_bndry: The boundary condition residual of shape (n_inputs, 1)
    """
    
    tx_equation = inputs[0]
    u_eqn, du_dt, du_dx, du_dxx = self.input_gradient(tx_equation)
    burgers_eq = du_dt + u_eqn*du_dx - self.nu*du_dxx


    tx_init = inputs[1]
    tx_bound = inputs[2]
    n_i = tx_init.shape[0]
    u_ib = self.network(tf.concat([tx_init, tx_bound], axis = 0))
    u_init = u_ib[:n_i]
    u_bound = u_ib[n_i:]

    return burgers_eq, u_init, u_bound 
  
  
  @staticmethod
  def build_network(layers, n_inputs=2, n_outputs=1, activation=keras.activations.tanh, initialization=keras.initializers.glorot_normal):
    """
    Builds a fully connected neural network with the specified number of layers and nodes per layer.

    Args:
        layers (list): List of integers specifying the number of nodes in each layer.
        n_inputs (int): Number of inputs to the network.
        n_outputs (int): Number of outputs from the network.
        activation (function): Activation function to use in each layer.
        initialization (function): Initialization function to use in each layer.
    returns:
        keras.Model: A keras model representing the neural network.
    """
    inputs = keras.layers.Input((n_inputs))
    x = inputs
    for i in layers:
      x = keras.layers.Dense(i, activation = activation, kernel_initializer=initialization)(x)
    
    outputs = keras.layers.Dense(n_outputs, kernel_initializer=initialization)(x)
    return keras.Model(inputs=[inputs], outputs = [outputs])


# Review the model and related utils function: Not sure about boundary and initial conditions
class WavePinn(keras.Model):
  """
  PINN model for the wave equation with Dirichlet boundary conditions.
  """

  def __init__(self, network, dimension, c) -> None:
    super().__init__()
    self.network = network
    self.dimension = dimension
    self.c = c

  @tf.function
  def input_diagonal_hessian(self, x):
    """
    computes the diagonal of the Hessian of the network output with respect to the inputs.
    Args:
      x: input tensor of shape (n_inputs, 2)
    Returns:
      u_tt: second derivative of u with respect to t
      u_xx: second derivative of u with respect to x
    """

    with tf.GradientTape() as g2tape:
      g2tape.watch(x)
      with tf.GradientTape() as gtape:
        gtape.watch(x)
        u = self.network(x)
      first_order = gtape.batch_jacobian(u, x)

    hessian = g2tape.batch_jacobian(first_order, x)
    u_tt = hessian[..., 0, 0]
    u_xx = hessian[..., 1, 1]
    return u_tt, u_xx

  
  @tf.function
  def input_gradient(self, x):
    """
    Compute the first derivative of the network output with respect to the inputs.
    Args:
      x: input tensor of shape (n_inputs, 2)
    Returns:
      u: network output of shape (n_inputs, 1)
      u_t: first derivative of u with respect to t
    """

    with tf.GradientTape() as g:
      g.watch(x)
      u = self.network(x)
    du_dt = g.batch_jacobian(u, x)[..., 0]
    return u, du_dt

  
  def call(self, inputs):
    """
    Performs forward pass of the model, computing the PDE residual and the initial and boundary conditions.

    Args:
      inputs: A list of tensors, where the first tensor is the equation data,
        the second tensor is the phi and psi initial condition data, and the third tensor
        is the boundary condition data.

    Returns:
        pde_residual: The PDE residual of shape (n_inputs, 1)
        u_phi: The phi initial condition output of shape (n_inputs, 1)
        du_dt_psi: The psi initial condition output of shape (n_inputs, 1)
        u_bndry: The boundary condition output of shape (n_inputs, 1)

    """

    tx_equation = inputs[0]
    tx_init = inputs[1]
    tx_bound = inputs[2]

    d2u_dt2, d2u_dx2 = self.input_gradient(tx_equation)

    
    # Calculate PDE residual
    pde_residual = d2u_dt2 - (self.c**2) * d2u_dx2

    u_phi, du_dt_psi = self.input_gradient(tx_init)

    u_bound = self.network(tx_bound)

    return tf.stack([pde_residual, u_phi, du_dt_psi, u_bound], axis=0)

  
  def fit(self, inputs, labels, epochs, optimizer, progress_interval=500) -> dict[str, list[float]]:
    """
    Train the model with the given inputs and optimizer.

    Args:
      inputs: A list of tensors, where the first tensor is the equation data,
        the second tensor is the initial condition data, and the fourth tensor is the
        boundary condition data.
      labels: A list of tensors, where the first tensor is the phi initial condition labels,
        the second tensor is the psi initial condition labels, and the third tensor is the
        boundary condition labels.
      epochs: The number of epochs to train for.
      optimizer : The optimizer to use for training.
      progress_interval: The number of epochs between each progress report.
    Returns:
      A dictionary containing the loss history for each of the three loss terms.
    """
    history = _create_history_dict()
    start_time = time.time()
    for epoch in range(epochs):
      with tf.GradientTape() as tape:
        residual, u_phi, du_dt_psi, u_bndry = self.call(inputs)

        loss_equation = tf.reduce_mean(tf.square(residual))
        loss_initial = tf.reduce_mean(tf.square(u_phi - labels[0])) + tf.reduce_mean(tf.square(du_dt_psi - labels[1]))
        loss_boundary = tf.reduce_mean(tf.square(u_bndry - labels[2]))
        loss = loss_equation + loss_initial + loss_boundary

      grads = tape.gradient(loss, self.trainable_weights)
      optimizer.apply_gradients(zip(grads, self.trainable_weights))

      _add_to_history_dict(history, loss, loss_equation, loss_initial, loss_boundary)
      
      if epoch % progress_interval == 0:
        print(f"Epoch: {epoch} Loss: {loss.numpy():.4f} Total Elapsed Time: {time.time() - start_time:.2f}")

    return history

  
  @staticmethod 
  def build_network(layers, n_inputs=2, n_outputs=1, activation=keras.activations.tanh, initialization=keras.initializers.glorot_normal):
    """
    Builds a fully connected neural network with the specified number of layers and nodes per layer.

    Args:
        layers (list): List of integers specifying the number of nodes in each layer.
        n_inputs (int): Number of inputs to the network.
        n_outputs (int): Number of outputs from the network.
        activation (function): Activation function to use in each layer.
        initialization (function): Initialization function to use in each layer.
    returns:
        keras.Model: A keras model representing the neural network.
    """
    inputs = keras.layers.Input((n_inputs))
    x = inputs
    for i in layers:
      x = keras.layers.Dense(i, activation = activation, kernel_initializer=initialization)(x)
    
    outputs = keras.layers.Dense(n_outputs, kernel_initializer=initialization)(x)
    return keras.Model(inputs=[inputs], outputs = [outputs])


class HeatPinn(keras.Model):
  """
  Keras PINN model for the Heat PDE.
  Attributes:
    network (keras.Model): The neural network used to approximate the solution.
    k (float): The thermal conductivity of the material.
  """

  def __init__(self, network: "keras.Model", k: float = 1.0) -> None:
    """
    Args:
      network: A keras model representing the backbone neural network.
      k: thermal conductivity. Default is 1.
    """
    super().__init__()
    self.network = network
    self.k = k


  def fit(self, inputs, labels, epochs, optimizer, progress_interval=500) -> dict[str, list[float]]:
    """
    Train the model with the given inputs and optimizer.

    Args:
      inputs: A list of tensors, where the first tensor is the equation data,
        the second tensor is the initial condition data, and the fourth tensor is the
        boundary condition data.
      labels: A list of tensors, where the first tensor is the phi initial condition labels,
        the second tensor is the boundary condition labels.
      epochs: The number of epochs to train for.
      optimizer : The optimizer to use for training.
      progress_interval: The number of epochs between each progress report.
    Returns:
      A dictionary containing the loss history for each of the three loss terms.
    """
    history = _create_history_dict()

    start_time = time.time()
    for epoch in range(epochs):
      with tf.GradientTape() as tape:
        residual, u_init, u_bndry = self.call(inputs)

        loss_residual = tf.reduce_mean(tf.square(residual))
        loss_init = tf.reduce_mean(tf.square(u_init - labels[0]))
        loss_boundary = tf.reduce_mean(tf.square(u_bndry - labels[1]))
        loss = loss_residual + loss_init + loss_boundary

      grads = tape.gradient(loss, self.trainable_weights)
      optimizer.apply_gradients(zip(grads, self.trainable_weights))

      _add_to_history_dict(history, loss, loss_residual, loss_init, loss_boundary)
      
      if epoch % progress_interval == 0:
        print(f"Epoch: {epoch} Loss: {loss.numpy():.4f} Total Elapsed Time: {time.time() - start_time:.2f}")
    
    return history

  
  @tf.function
  def input_gradient(self, tx):
    """
    Compute the first order derivative w.r.t. time and second order derivative w.r.t. space of the network output.

    Args:
      tx: input tensor of shape (n_inputs, 2)

    Returns:
      u_t: first derivative of u with respect to t
      u_xx: second derivative of u with respect to x
    """
    with tf.GradientTape() as gg:
      gg.watch(tx)
      with tf.GradientTape() as g:
        g.watch(tx)
        u = self.network(tx)

      first_order = g.batch_jacobian(u, tx)
      du_dt = first_order[..., 0]
      du_dx = first_order[..., 1]

    d2u_dx2 = gg.batch_jacobian(du_dx, tx)[..., 1]

    return du_dt, d2u_dx2
    

  
  def call(self, inputs):
    """
    Performs forward pass of the model, computing the PDE residual and the initial and boundary conditions.

    Args:
      inputs: A list of tensors, where the first tensor is the equation data,
        the second tensor is the initial condition data, and the third tensor is the
        boundary condition data.

    Returns:
        pde_residual: The PDE residual of shape (n_inputs, 1)
        u_init: The initial condition output of shape (n_inputs, 1)
        u_bndry: The boundary condition output of shape (n_inputs, 1)

    """

    tx_equation = inputs[0]
    tx_init = inputs[1]
    tx_bound = inputs[2]

    du_dt, d2u_dx2 = self.input_gradient(tx_equation)

    
    # Calculate PDE residual
    pde_residual = du_dt - (self.k) * d2u_dx2

    n_i = tf.shape(tx_init)[0]
    tx_ib = tf.concat([tx_init, tx_bound], axis=0)
    u_ib = self.network(tx_ib)
    u_init = u_ib[:n_i]
    u_bndry = u_ib[n_i:]

    return pde_residual, u_init, u_bndry

  @staticmethod
  def build_network(layers, n_inputs=2, n_outputs=1, activation=keras.activations.tanh, initialization=keras.initializers.glorot_normal):
    """
    Builds a fully connected neural network with the specified number of layers and nodes per layer.

    Args:
        layers (list): List of integers specifying the number of nodes in each layer.
        n_inputs (int): Number of inputs to the network.
        n_outputs (int): Number of outputs from the network.
        activation (function): Activation function to use in each layer.
        initialization (function): Initialization function to use in each layer.
    returns:
        keras.Model: A keras model representing the neural network.
    """
    inputs = keras.layers.Input((n_inputs))
    x = inputs
    for i in layers:
      x = keras.layers.Dense(i, activation = activation, kernel_initializer=initialization)(x)
    
    outputs = keras.layers.Dense(n_outputs, kernel_initializer=initialization)(x)
    return keras.Model(inputs=[inputs], outputs = [outputs])


class SchroedingerPinn(keras.Model):
  """
  Keras PINN model for the Schroedinger PDE.
  Attributes:
    network (keras.Model): The neural network used to approximate the solution.
    k (float): The planck constant. Default is 0.5.
  """

  def __init__(self, network: "keras.Model", k: float = 0.5) -> None:
    super().__init__()
    self.network = network
    self.k = k


  def fit(self, inputs, labels, epochs, optimizer, n_boundary_samples, progress_interval=500) -> dict[str, list[float]]:
    """
    Train the model with the given inputs and optimizer.
    Args:
      inputs: A list of tensors, where the first tensor is the equation data, the second tensor is the initial condition data, and the third tensor is the boundary condition data.
      labels: A list of tensors, where the first tensor is the initial condition labels, the second tensor is the initial condition labels, and the third tensor is the boundary condition labels.
      epochs: The number of epochs to train for.
      optimizer : The optimizer to use for training.
      n_boundary_samples: The number of boundary samples to use for training.
      progress_interval: The number of epochs between each progress report.
    Returns:
      A dictionary containing the loss history for each of the three loss terms.
    """

    history = _create_history_dict()

    start_time = time.time()
    for epoch in range(epochs):
      with tf.GradientTape() as tape:
        residual, h_init, h_bndry, dhb_dx = self.call(inputs)

        loss_residual = tf.reduce_mean(tf.abs(residual))
        loss_init = tf.reduce_mean(tf.reduce_sum(tf.square(h_init - labels[0]), axis=1))
        loss_boundary = tf.reduce_mean(tf.reduce_sum(tf.square(h_bndry[:n_boundary_samples//2] - h_bndry[n_boundary_samples//2:]), axis=1)) + tf.reduce_mean(tf.reduce_sum(tf.square(dhb_dx[:n_boundary_samples//2] - dhb_dx[n_boundary_samples//2:]), axis=1))
        loss = loss_residual + loss_init + loss_boundary

      grads = tape.gradient(loss, self.trainable_weights)
      optimizer.apply_gradients(zip(grads, self.trainable_weights))

      _add_to_history_dict(history, loss_residual, loss_init, loss_boundary)
      
      if epoch % progress_interval == 0:
        print(f"Epoch: {epoch} Loss: {loss.numpy():.4f} Total Elapsed Time: {time.time() - start_time:.2f}")
      
    return history


  @tf.function
  def input_gradient_equation(self, tx):
    """
    Compute the first order derivative w.r.t. time and second order derivative w.r.t. space of the network output.
    Args:
      tx: input tensor of shape (n_inputs, 2)
    Returns:
      h: network output of shape (n_inputs, 2)
      dh_dt: first derivative of h with respect to t of shape (n_inputs, 2). The first column is the real part derivative and the second column is the imaginary part derivative.
      d2h_dx2: second derivative of h with respect to x of shape (n_inputs, 2). The first column is the real part derivative and the second column is the imaginary part derivative.
    """

    with tf.GradientTape() as gg:
      gg.watch(tx)
      
      with tf.GradientTape() as g:
        g.watch(tx)
        h = self.network(tx) # first column is real part, second column is imaginary part
      
      first_order = g.batch_jacobian(h, tx)
      dh_dt = first_order[:, :, 0]
      dh_dx = first_order[:, :, 1]


    d2h_dx2 = gg.batch_jacobian(dh_dx, tx)[:, :, 1]

    return h, dh_dt, d2h_dx2


  @tf.function
  def input_gradient_boundary(self, tx):
    """
    Compute the first order derivative w.r.t. space of the network output.
    Args:
      tx: input tensor of shape (n_inputs, 2)
    Returns:
      h: network output of shape (n_inputs, 2)
      dh_dx: first derivative of h with respect to x of shape (n_inputs, 2). The first column is the real part derivative and the second column is the imaginary part derivative.

    """

    with tf.GradientTape() as g:
      g.watch(tx)
      h = self.network(tx)
    
    dh_dx = g.batch_jacobian(h, tx)[:, :, 1]

    return h, dh_dx


  def call(self, inputs):
    tx_equation = inputs[0]
    tx_init = inputs[1]
    tx_bound = inputs[2]

    h, dh_dt, d2h_dx2 = self.input_gradient_equation(tx_equation)

    # Calculate PDE residual
    h = tf.complex(h[:, 0:1], h[:, 1:2])
    dh_dt = tf.complex(dh_dt[:, 0:1], dh_dt[:, 1:2])
    d2h_dx2 = tf.complex(d2h_dx2[:, 0:1], d2h_dx2[:, 1:2])
    pde_residual = 1j * dh_dt + self.k * d2h_dx2 + (h * tf.math.conj(h)) * h

    h_init = self.network(tx_init)

    h_bound, dhb_dx = self.input_gradient_boundary(tx_bound)

    return pde_residual, h_init, h_bound, dhb_dx

  
  @staticmethod
  def build_network(layers, n_inputs=2, n_outputs=2, activation=keras.activations.tanh, initialization=keras.initializers.glorot_normal):
    """
    Builds a fully connected neural network with the specified number of layers and nodes per layer. The network outputs the real and imaginary parts of the solution.

    Args:
        layers (list): List of integers specifying the number of nodes in each layer.
        n_inputs (int): Number of inputs to the network. Default is 2.
        n_outputs (int): Number of outputs from the network. Default is 2.
        activation (function): Activation function to use in each layer.
        initialization (function): Initialization function to use in each layer.
    returns:
        keras.Model: A keras model representing the neural network.
    """
    inputs = keras.layers.Input((n_inputs))
    x = inputs
    for i in layers:
      x = keras.layers.Dense(i, activation = activation, kernel_initializer=initialization)(x)
    
    outputs = keras.layers.Dense(n_outputs, kernel_initializer=initialization)(x)
    return keras.Model(inputs=[inputs], outputs = [outputs])


class PoissonPinn(keras.Model):
  
  def __init__(self, network) -> None:
    super().__init__()
    self.network = network


  def fit(self, inputs, labels, epochs, optimizer, progress_interval=500) -> dict[str, list[float]]:

    history = _create_history_dict()
    start_time = time.time()
    for epoch in range(epochs):
      with tf.GradientTape() as tape:
        d2u_dx2, u_bndry = self.call(inputs)

        loss_residual = tf.reduce_mean(tf.square(d2u_dx2 - labels[0]))
        loss_boundary = tf.reduce_mean(tf.square(u_bndry - labels[1]))
        loss = loss_residual + loss_boundary
      
      grads = tape.gradient(loss, self.trainable_weights)
      optimizer.apply_gradients(zip(grads, self.trainable_weights))

      _add_to_history_dict(history, loss_residual, None, loss_boundary)

      if epoch % progress_interval == 0:
        print(f"Epoch: {epoch} Loss: {loss.numpy():.4f} Total Elapsed Time: {time.time() - start_time:.2f}")

    return history

  
  def call(self, inputs):
    x_equation = inputs[0]
    x_boundary = inputs[1]

    d2u_dx2 = self.laplace(x_equation)
    u_boundary = self.network(x_boundary)

    return d2u_dx2, u_boundary


  @tf.function
  def laplace(self, x):
    with tf.GradientTape() as gg:
      gg.watch(x)
      
      with tf.GradientTape() as g:
        g.watch(x)
        u = self.network(x)
      
      du_dx = g.batch_jacobian(u, x)[:, 0]

    d2u_dx2 = gg.batch_jacobian(du_dx, x)[:, 0]

    return d2u_dx2


  @staticmethod
  def build_network(layers, n_inputs=1, n_outputs=1, activation=keras.activations.tanh, initialization=keras.initializers.glorot_normal):
    """
    Builds a fully connected neural network with the specified number of layers and nodes per layer.

    Args:
        layers (list): List of integers specifying the number of nodes in each layer.
        n_inputs (int): Number of inputs to the network. Default is 2.
        n_outputs (int): Number of outputs from the network. Default is 2.
        activation (function): Activation function to use in each layer.
        initialization (function): Initialization function to use in each layer.
    returns:
        keras.Model: A keras model representing the neural network.
    """
    inputs = keras.layers.Input((n_inputs))
    x = inputs
    for i in layers:
      x = keras.layers.Dense(i, activation = activation, kernel_initializer=initialization)(x)
    
    outputs = keras.layers.Dense(n_outputs, kernel_initializer=initialization)(x)
    return keras.Model(inputs=[inputs], outputs = [outputs])

from tensorflow import keras
import tensorflow as tf
import numpy as np
import time



class BurgersPinn(keras.Model):
  def __init__(self, network, nu, n_inputs=2, n_outputs=1):
    super().__init__()
    self.network = network
    self.nu = nu

  def fit(self, inputs, labels, epochs, optimizer, progress_interval=500):
    """
    Train the model with the given inputs and optimizer.

    Args:
      inputs: A list of tensors, where the first tensor is the equation data,
        the second tensor is the initial condition data, and the third tensor
        is the boundary condition data.
      epochs: The number of epochs to train for.
      optimizer : The optimizer to use for training.
      progress_interval: The number of epochs between each progress report.
    """
    start_time = time.time()
    for epoch in range(epochs):
      with tf.GradientTape() as tape:
        burgers_eq, u_init, u_bndry = self.call(inputs)
        loss = tf.reduce_mean(burgers_eq**2) + tf.reduce_mean(tf.square(u_init - labels[1])) + tf.reduce_mean(tf.square(u_bndry - labels[2]))

      grads = tape.gradient(loss, self.trainable_weights)
      optimizer.apply_gradients(zip(grads, self.trainable_weights))
      if epoch % progress_interval == 0:
        print(f"Epoch: {epoch} Loss: {loss.numpy():.4f} Total Elapsed Time: {time.time() - start_time:.2f}")

  
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
'''
WIP v2 of the models. This is a work in progress and is not stable yet.
In the future, this will be the main model file.
This implementation will try to use the tf.keras API as much as possible and fix optimization issues with the current implementation.
'''

from typing import Tuple, List
import tensorflow as tf
import numpy as np
from modules.utils import LOSS_BOUNDARY, LOSS_INITIAL, LOSS_RESIDUAL, MEAN_ABSOLUTE_ERROR

def create_dense_model(layers: List[int], activation: "tf.keras.activations.Activation", \
    initializer: "tf.keras.initializers.Initializer", n_inputs: int, n_outputs: int, **kwargs) -> "tf.keras.Model":
    """
    Creates a dense model with the given layers, activation, and input and output sizes.

    Args:
        layers: The sizes of the hidden layers.
        activation: The activation function to use.
        initializer: The initializer to use.
        n_inputs: The number of inputs.
        n_outputs: The number of outputs.
        **kwargs: Additional arguments to pass to the Model constructor.

    Returns:
        The dense model.
    """
    inputs = tf.keras.Input(shape=(n_inputs,))
    x = inputs
    for layer in layers:
        x = tf.keras.layers.Dense(layer, activation=activation, kernel_initializer=initializer)(x)
    outputs = tf.keras.layers.Dense(n_outputs, kernel_initializer=initializer)(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, **kwargs)


class AdvectionPinn(tf.keras.Model):
    """
    A PINN for the advection equation.

    Attributes:
        backbone: The backbone model.
        v: The velocity of the advection.
        k: The diffusion coefficient.
        loss_boundary_tracker: The boundary loss tracker.
        loss_residual_tracker: The residual loss tracker.
        mae_tracker: The mean absolute error tracker.
        loss_boundary_weight: The weight of the boundary loss.
        loss_residual_weight: The weight of the residual loss.

    """

    def __init__(self, backbone, v: float, k: float, loss_residual_weight: float = 1.0, loss_boundary_weight: float = 1.0, **kwargs):
        """
        Initializes the model.

        Args:
            backbone: The backbone model.
            v: The velocity of the advection.
            k: The diffusion coefficient.
            loss_residual_weight: The weight of the residual loss.
            loss_boundary_weight: The weight of the boundary loss.
            **kwargs: Additional arguments to pass to the Model constructor.
        """
        super().__init__(**kwargs)
        self.backbone = backbone
        self.v = v
        self.k = k
        self.loss_boundary_tracker = tf.keras.metrics.Mean(name=LOSS_BOUNDARY)
        self.loss_residual_tracker = tf.keras.metrics.Mean(name=LOSS_RESIDUAL)
        self.mae_tracker = tf.keras.metrics.MeanAbsoluteError(name=MEAN_ABSOLUTE_ERROR)
        self._loss_residual_weight = tf.Variable(loss_residual_weight, trainable=False, name="loss_residual_weight")
        self._loss_boundary_weight = tf.Variable(loss_boundary_weight, trainable=False, name="loss_boundary_weight")


    def set_loss_weights(self, loss_residual_weight: float, loss_boundary_weight: float):
        """
        Sets the loss weights.

        Args:
            loss_residual_weight: The weight of the residual loss.
            loss_boundary_weight: The weight of the boundary loss.
        """
        self._loss_residual_weight.assign(loss_residual_weight)
        self._loss_boundary_weight.assign(loss_boundary_weight)

    @tf.function
    def call(self, inputs: "tf.Tensor", training: bool = False) -> "tf.Tensor":
        """
        Calls the model on the inputs.

        Args:
            inputs: The inputs to the model. First input is the samples, second input is the boundary.
            training: Whether or not the model is being called in training mode.

        Returns:
            The output of the model.
        """
        
        #compute the derivatives
        inputs_residuals = inputs[0]
        inputs_bnd = inputs[1]

        with tf.GradientTape(watch_accessed_variables=False) as tape2:
            tape2.watch(inputs_residuals)
            with tf.GradientTape(watch_accessed_variables=False) as tape1:
                tape1.watch(inputs_residuals)
                u_samples = self.backbone(inputs_residuals, training=training)
            u_x = tape1.gradient(u_samples, inputs_residuals)
        u_xx = tape2.gradient(u_x, inputs_residuals)

        #compute the lhs
        lhs_samples = self.v * u_x - self.k * u_xx

        #compute the boundary
        u_bnd = self.backbone(inputs_bnd, training=training)

        return u_samples, lhs_samples, u_bnd

    def train_step(self, data: Tuple["tf.Tensor", "tf.Tensor"]) -> "tf.Tensor":
        """
        Performs a training step on the given data.

        Args:
            data: The data to train on. First data is the inputs,second data is the outputs.\
                In inputs, first tensor is the residual samples, second tensor is the boundary samples.\
                    In outputs, first tensor is the exact u for the residual samples, second tensor is the \
                        exact rhs for the residual samples, and third tensor is the exact u for the boundary samples.

        Returns:
            The loss.
        """
        x, y = data

        # compute residual loss with samples
        with tf.GradientTape() as tape:
            u_samples, lhs_samples, u_bnd = self(x, training=True)
            loss_residual = tf.losses.mean_squared_error(y[1], lhs_samples)
            loss_boundary = tf.losses.mean_squared_error(y[2], u_bnd)
            loss = self._loss_residual_weight * loss_residual + self._loss_boundary_weight * loss_boundary

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.loss_residual_tracker.update_state(loss_residual)
        self.loss_boundary_tracker.update_state(loss_boundary)
        self.mae_tracker.update_state(y[0], u_samples)


        return {m.name: m.result() for m in self.metrics}
    
    @property
    def metrics(self):
        '''
        Returns the metrics of the model.
        '''
        return [self.loss_boundary_tracker, self.loss_residual_tracker, self.mae_tracker]


    




    

    


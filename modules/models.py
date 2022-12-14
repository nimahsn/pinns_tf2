'''
This file contains the PINN models for the Advection, Burgers, Schrodinger, Poisson, Heat, and Wave equations.
'''

from typing import Tuple, List, Union
import tensorflow as tf
import numpy as np

LOSS_BOUNDARY = "loss_boundary"
LOSS_INITIAL = "loss_initial"
LOSS_RESIDUAL = "loss_residual"
MEAN_ABSOLUTE_ERROR = "mean_absolute_error"

def create_dense_model(layers: List[Union[int, "tf.keras.layers.Layer"]], activation: "tf.keras.activations.Activation", \
    initializer: "tf.keras.initializers.Initializer", n_inputs: int, n_outputs: int, **kwargs) -> "tf.keras.Model":
    """
    Creates a dense model with the given layers, activation, and input and output sizes.

    Args:
        layers: The layers to use. Elements can be either an integer or a Layer instance. If an integer, a Dense layer with that many units will be used.
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
        if isinstance(layer, int):
            x = tf.keras.layers.Dense(layer, activation=activation, kernel_initializer=initializer)(x)
        else:
            x = layer(x)
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
        self._loss_residual_weight = tf.Variable(loss_residual_weight, trainable=False, name="loss_residual_weight", dtype=tf.keras.backend.floatx())
        self._loss_boundary_weight = tf.Variable(loss_boundary_weight, trainable=False, name="loss_boundary_weight", dtype=tf.keras.backend.floatx())


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


class PoissonPinn(tf.keras.Model):
    """
    A PINN for the Poisson's equation.
    
    Attributes:
        backbone: The backbone model.
        loss_boundary_tracker: The boundary loss tracker.
        loss_residual_tracker: The residual loss tracker.
        mae_tracker: The mean absolute error tracker.
        _loss_residual_weight: The weight of the residual loss.
        _loss_boundary_weight: The weight of the boundary loss.
    """
    
    def __init__(self, backbone, loss_residual_weight: float = 1.0, loss_boundary_weight: float = 1.0, **kwargs):
        """
        Initializes the model.

        Args:
            backbone: The backbone model.
            loss_residual_weight: The weight of the residual loss.
            loss_boundary_weight: The weight of the boundary loss.
            **kwargs: Additional arguments to pass to the Model constructor.
        """
        super().__init__(**kwargs)
        self.backbone = backbone
        self.loss_boundary_tracker = tf.keras.metrics.Mean(name=LOSS_BOUNDARY)
        self.loss_residual_tracker = tf.keras.metrics.Mean(name=LOSS_RESIDUAL)
        self.mae_tracker = tf.keras.metrics.MeanAbsoluteError(name=MEAN_ABSOLUTE_ERROR)
        self._loss_residual_weight = tf.Variable(loss_residual_weight, trainable=False, name="loss_residual_weight", dtype=tf.keras.backend.floatx())
        self._loss_boundary_weight = tf.Variable(loss_boundary_weight, trainable=False, name="loss_boundary_weight", dtype=tf.keras.backend.floatx())

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
    def call(self, inputs, training=False):
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
            u_x = tape1.batch_jacobian(u_samples, inputs_residuals)[:, :, 0]
        u_xx = tape2.batch_jacobian(u_x, inputs_residuals)[:, :, 0]

        #compute the lhs
        lhs_samples = u_xx

        #compute the boundary
        u_bnd = self.backbone(inputs_bnd, training=training)

        return u_samples, lhs_samples, u_bnd

    def train_step(self, data):
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
        inputs, outputs = data
        u_exact, rhs_exact, u_bnd_exact = outputs

        # compute residual loss with samples
        with tf.GradientTape() as tape:
            u_samples, lhs_samples, u_bnd = self(inputs, training=True)
            loss_residual = tf.losses.mean_squared_error(rhs_exact, lhs_samples)
            loss_boundary = tf.losses.mean_squared_error(u_bnd_exact, u_bnd)
            loss = self._loss_residual_weight * loss_residual + self._loss_boundary_weight * loss_boundary

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.loss_residual_tracker.update_state(loss_residual)
        self.loss_boundary_tracker.update_state(loss_boundary)
        self.mae_tracker.update_state(u_exact, u_samples)

        return {m.name: m.result() for m in self.metrics}

    
    @property
    def metrics(self):
        '''
        Returns the metrics of the model.
        '''
        return [self.loss_boundary_tracker, self.loss_residual_tracker, self.mae_tracker]


class SchrodingerPinn(tf.keras.Model):
    """
    A PINN for the Schrodinger's equation.
    
    Attributes:
        backbone: The backbone model.
        k: The planck constant.
        loss_boundary_tracker: The boundary loss tracker.
        loss_initial_tracker: The initial loss tracker.
        loss_residual_tracker: The residual loss tracker.
        mae_tracker: The mean absolute error tracker.
        _loss_residual_weight: The weight of the residual loss.
        _loss_initial_weight: The weight of the initial loss.
        _loss_boundary_weight: The weight of the boundary loss.
    """
    
    def __init__(self, backbone, k: float = 0.5, loss_residual_weight: float = 1.0, loss_initial_weight: float = 1.0, \
        loss_boundary_weight: float = 1.0, **kwargs):
        """
        Initializes the model.

        Args:
            backbone: The backbone model.
            k: The planck constant.
            loss_residual_weight: The weight of the residual loss.
            loss_initial_weight: The weight of the initial loss.
            loss_boundary_weight: The weight of the boundary loss.
            **kwargs: Additional arguments to pass to the Model constructor.
        """
        super().__init__(**kwargs)
        self.backbone = backbone
        self.k = k
        self.loss_boundary_tracker = tf.keras.metrics.Mean(name=LOSS_BOUNDARY)
        self.loss_initial_tracker = tf.keras.metrics.Mean(name=LOSS_INITIAL)
        self.loss_residual_tracker = tf.keras.metrics.Mean(name=LOSS_RESIDUAL)
        self.mae_tracker = tf.keras.metrics.MeanAbsoluteError(name=MEAN_ABSOLUTE_ERROR)
        self._loss_residual_weight = tf.Variable(loss_residual_weight, trainable=False, name="loss_residual_weight", dtype=tf.keras.backend.floatx())
        self._loss_boundary_weight = tf.Variable(loss_boundary_weight, trainable=False, name="loss_boundary_weight", dtype=tf.keras.backend.floatx())
        self._loss_initial_weight = tf.Variable(loss_initial_weight, trainable=False, name="loss_initial_weight", dtype=tf.keras.backend.floatx())


    def set_loss_weights(self, loss_residual_weight: float, loss_initial_weight: float, loss_boundary_weight: float):
        """
        Sets the loss weights.

        Args:
            loss_residual_weight: The weight of the residual loss.
            loss_initial_weight: The weight of the initial loss.
            loss_boundary_weight: The weight of the boundary loss.
        """
        self._loss_residual_weight.assign(loss_residual_weight)
        self._loss_boundary_weight.assign(loss_boundary_weight)
        self._loss_initial_weight.assign(loss_initial_weight)

    @tf.function
    def call(self, inputs, training=False):
        """
        Calls the model on the inputs.

        Args:
            inputs: The inputs to the model. First input is the samples, second input is the initial, \
                third input is the boundary start, fourth input is the boundary end.
            training: Whether or not the model is being called in training mode.

        Returns:
            The output of the model. First output is the solution for residual samples, second is the lhs residual, \
                third is solution for initial samples, fourth is solution for boundary samples, and fifth is dh/dx for boundary samples.
        """
        inputs_residuals = inputs[0]
        inputs_initial = inputs[1]
        inputs_bnd_start = inputs[2]
        inputs_bnd_end = inputs[3]

        with tf.GradientTape(watch_accessed_variables=False) as tape2:
            tape2.watch(inputs_residuals)
            with tf.GradientTape(watch_accessed_variables=False) as tape1:
                tape1.watch(inputs_residuals)
                h_samples = self.backbone(inputs_residuals, training=training)
            
            first_order = tape1.batch_jacobian(h_samples, inputs_residuals) # output is n_sample x 2 * 2
            dh_dt = first_order[:, :, 0]
            dh_dx = first_order[:, :, 1]
        d2h_dx2 = tape2.batch_jacobian(dh_dx, inputs_residuals)[:, :, 1]

        norm2_h = h_samples[:, 0:1] ** 2 + h_samples[:, 1:2] ** 2
        real_residual = -dh_dt[:, 1:2] + self.k * d2h_dx2[:, 0:1] + norm2_h * h_samples[:, 0:1]
        imag_residual = dh_dt[:, 0:1] + self.k * d2h_dx2[:, 1:2] + norm2_h * h_samples[:, 1:2]

        lhs_samples = tf.concat([real_residual, imag_residual], axis=1)

        h_initial = self.backbone(inputs_initial, training=training)

        inputs_bnd = tf.concat([inputs_bnd_start, inputs_bnd_end], axis=0)

        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(inputs_bnd)
            h_bnd = self.backbone(inputs_bnd, training=training)
        dh_dx_bnd = tape.batch_jacobian(h_bnd, inputs_bnd)[:, :, 1]

        h_bnd_start = h_bnd[0:inputs_bnd_start.shape[0]]
        h_bnd_end = h_bnd[inputs_bnd_start.shape[0]:]
        dh_dx_start = dh_dx_bnd[0:inputs_bnd_start.shape[0]]
        dh_dx_end = dh_dx_bnd[inputs_bnd_start.shape[0]:]

        return h_samples, lhs_samples, h_initial, h_bnd_start, h_bnd_end, dh_dx_start, dh_dx_end

    @tf.function
    def train_step(self, data):
        """
        Performs a training step.

        Args:
            data: The data to train on. First input is the samples, second input is the initial, third input is the boundary. \
                First output is the solution for residual samples, second is the lhs residual, third is solution for initial samples. \
                    The boundary samples must have 3 columns, where the first column is the t value, the second column is the x value for the start \
                        of the boundary, and the third column is the x value for the end of the boundary.

        Returns:
            The loss of the step.
        """
        inputs, outputs = data
        tx_samples, tx_initial, txx_bnd = inputs
        tx_bnd_start = tf.concat([txx_bnd[:, 0:1], txx_bnd[:, 1:2]], axis=1)
        tx_bnd_end = tf.concat([txx_bnd[:, 0:1], txx_bnd[:, 2:3]], axis=1)
        h_samples_exact, rhs_samples_exact, h_initial_exact = outputs

        with tf.GradientTape() as tape:
            h_samples, lhs_samples, h_initial, h_bnd_start, h_bnd_end, dh_dx_start, dh_dx_end = \
                self([tx_samples, tx_initial, tx_bnd_start, tx_bnd_end], training=True)

            loss_residual = tf.losses.mean_squared_error(rhs_samples_exact, lhs_samples)
            loss_initial = tf.losses.mean_squared_error(h_initial_exact, h_initial)
            loss_boundary_h = tf.losses.mean_squared_error(h_bnd_start, h_bnd_end)
            loss_boundary_dh_dx = tf.losses.mean_squared_error(dh_dx_start, dh_dx_end)
            loss_boundary = loss_boundary_h + loss_boundary_dh_dx
            loss = self._loss_residual_weight * loss_residual + self._loss_initial_weight * loss_initial + \
                self._loss_boundary_weight * loss_boundary

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.loss_residual_tracker.update_state(loss_residual)
        self.loss_initial_tracker.update_state(loss_initial)
        self.loss_boundary_tracker.update_state(loss_boundary)
        self.mae_tracker.update_state(h_samples_exact, h_samples)

        return {m.name: m.result() for m in self.metrics}

    @property
    def metrics(self):
        """
        Returns the metrics of the model.
        """
        return [self.loss_residual_tracker, self.loss_initial_tracker, self.loss_boundary_tracker, self.mae_tracker]


class BurgersPinn(tf.keras.Model):
    """
    A model that solves the Burgers' equation.
    """
    def __init__(self, backbone, nu: float, loss_residual_weight=1.0, loss_initial_weight=1.0, \
        loss_boundary_weight=1.0, **kwargs):
        """
        Initializes the model.

        Args:
            backbone: The backbone model.
            nu: The viscosity of the fluid.
            loss_residual_weight: The weight of the residual loss.
            loss_initial_weight: The weight of the initial loss.
            loss_boundary_weight: The weight of the boundary loss.
            **kwargs: Additional arguments.
        """
        super(BurgersPinn, self).__init__(**kwargs)

        self.backbone = backbone
        self.nu = nu
        self.loss_residual_tracker = tf.keras.metrics.Mean(name=LOSS_RESIDUAL)
        self.loss_initial_tracker = tf.keras.metrics.Mean(name=LOSS_INITIAL)
        self.loss_boundary_tracker = tf.keras.metrics.Mean(name=LOSS_BOUNDARY)
        self.mae_tracker = tf.keras.metrics.MeanAbsoluteError(name=MEAN_ABSOLUTE_ERROR)
        self._loss_residual_weight = tf.Variable(loss_residual_weight, trainable=False, name="loss_residual_weight", dtype=tf.keras.backend.floatx())
        self._loss_boundary_weight = tf.Variable(loss_boundary_weight, trainable=False, name="loss_boundary_weight", dtype=tf.keras.backend.floatx())
        self._loss_initial_weight = tf.Variable(loss_initial_weight, trainable=False, name="loss_initial_weight", dtype=tf.keras.backend.floatx())


    def set_loss_weights(self, loss_residual_weight: float, loss_initial_weight: float, loss_boundary_weight: float):
        """
        Sets the loss weights.

        Args:
            loss_residual_weight: The weight of the residual loss.
            loss_initial_weight: The weight of the initial loss.
            loss_boundary_weight: The weight of the boundary loss.
        """
        self._loss_residual_weight.assign(loss_residual_weight)
        self._loss_boundary_weight.assign(loss_boundary_weight)
        self._loss_initial_weight.assign(loss_initial_weight)

    @tf.function
    def call(self, inputs, training=False):
        """
        Performs a forward pass.

        Args:
            inputs: The inputs to the model. First input is the samples, second input is the initial, \
                and third input is the boundary data.
            training: Whether or not the model is training.

        Returns:
            The solution for the residual samples, the lhs residual, the solution for the initial samples, \
                and the solution for the boundary samples.
        """
        
        tx_samples = inputs[0]
        tx_init = inputs[1]
        tx_bnd = inputs[2]

        with tf.GradientTape(watch_accessed_variables=False) as tape2:
            tape2.watch(tx_samples)
            
            with tf.GradientTape(watch_accessed_variables=False) as tape1:
                tape1.watch(tx_samples)
                u_samples = self.backbone(tx_samples, training=training)

            first_order = tape1.batch_jacobian(u_samples, tx_samples)
            du_dt = first_order[..., 0]
            du_dx = first_order[..., 1]

        d2u_dx2 = tape2.batch_jacobian(du_dx, tx_samples)[..., 1]

        lhs_samples = du_dt + u_samples * du_dx - self.nu * d2u_dx2
        tx_ib = tf.concat([tx_init, tx_bnd], axis=0)
        u_ib = self.backbone(tx_ib, training=training)
        u_initial = u_ib[:tx_init.shape[0]]
        u_bnd = u_ib[tx_init.shape[0]:]

        return u_samples, lhs_samples, u_initial, u_bnd

    @tf.function
    def train_step(self, data):
        """
        Performs a training step.

        Args:
            data: The data to train on. First input is the samples, second input is the initial, \
                and third input is the boundary data. First output is the exact solution for the samples, \
                second output is the exact rhs for the samples, third output is the exact solution for the initial, \
                and fourth output is the exact solution for the boundary.

        Returns:
            The metrics for the training step.
        """

        inputs, outputs = data
        u_samples_exact, rhs_samples_exact, u_initial_exact, u_bnd_exact = outputs

        with tf.GradientTape() as tape:
            u_samples, lhs_samples, u_initial, u_bnd = self(inputs, training=True)

            loss_residual = tf.losses.mean_squared_error(rhs_samples_exact, lhs_samples)
            loss_initial = tf.losses.mean_squared_error(u_initial_exact, u_initial)
            loss_boundary = tf.losses.mean_squared_error(u_bnd_exact, u_bnd)
            loss = self._loss_residual_weight * loss_residual + self._loss_initial_weight * loss_initial + \
                self._loss_boundary_weight * loss_boundary

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.mae_tracker.update_state(u_samples_exact, u_samples)
        self.loss_residual_tracker.update_state(loss_residual)
        self.loss_initial_tracker.update_state(loss_initial)
        self.loss_boundary_tracker.update_state(loss_boundary)

        return {m.name: m.result() for m in self.metrics}

    @property
    def metrics(self):
        """
        Returns the metrics of the model.
        """
        return [self.loss_residual_tracker, self.loss_initial_tracker, self.loss_boundary_tracker, self.mae_tracker]


class HeatPinn(tf.keras.Model):
    """
    A model that solves the heat equation.
    """
    def __init__(self, backbone, k: float = 1.0, loss_residual_weight=1.0, loss_initial_weight=1.0, \
        loss_boundary_weight=1.0, **kwargs):
        """
        Initializes the model.

        Args:
            backbone: The backbone model.
            k: The heat diffusivity constant.
            loss_residual_weight: The weight of the residual loss.
            loss_initial_weight: The weight of the initial loss.
            loss_boundary_weight: The weight of the boundary loss.
            **kwargs: Additional keyword arguments.
        """
        super(HeatPinn, self).__init__(**kwargs)

        self.backbone = backbone
        self.k = k
        self.loss_residual_tracker = tf.keras.metrics.Mean(name=LOSS_RESIDUAL)
        self.loss_initial_tracker = tf.keras.metrics.Mean(name=LOSS_INITIAL)
        self.loss_boundary_tracker = tf.keras.metrics.Mean(name=LOSS_BOUNDARY)
        self.mae_tracker = tf.keras.metrics.MeanAbsoluteError(name=MEAN_ABSOLUTE_ERROR)
        self._loss_residual_weight = tf.Variable(loss_residual_weight, trainable=False, name="loss_residual_weight", dtype=tf.keras.backend.floatx())
        self._loss_boundary_weight = tf.Variable(loss_boundary_weight, trainable=False, name="loss_boundary_weight", dtype=tf.keras.backend.floatx())
        self._loss_initial_weight = tf.Variable(loss_initial_weight, trainable=False, name="loss_initial_weight", dtype=tf.keras.backend.floatx())


    def set_loss_weights(self, loss_residual_weight: float, loss_initial_weight: float, loss_boundary_weight: float):
        """
        Sets the loss weights.

        Args:
            loss_residual_weight: The weight of the residual loss.
            loss_initial_weight: The weight of the initial loss.
            loss_boundary_weight: The weight of the boundary loss.
        """
        self._loss_residual_weight.assign(loss_residual_weight)
        self._loss_boundary_weight.assign(loss_boundary_weight)
        self._loss_initial_weight.assign(loss_initial_weight)


    @tf.function
    def call(self, inputs, training=False):
        """
        Performs a forward pass.

        Args:
            inputs: The inputs to the model. First input is the samples, second input is the initial, \
                and third input is the boundary data.
            training: Whether or not the model is training.

        Returns:
            The solution for the residual samples, the lhs residual, the solution for the initial samples, \
                and the solution for the boundary samples.
        """
        
        tx_samples = inputs[0]
        tx_init = inputs[1]
        tx_bnd = inputs[2]

        with tf.GradientTape(watch_accessed_variables=False) as tape2:
            tape2.watch(tx_samples)
            
            with tf.GradientTape(watch_accessed_variables=False) as tape1:
                tape1.watch(tx_samples)
                u_samples = self.backbone(tx_samples, training=training)

            first_order = tape1.batch_jacobian(u_samples, tx_samples)
            du_dt = first_order[..., 0]
            du_dx = first_order[..., 1]
        d2u_dx2 = tape2.batch_jacobian(du_dx, tx_samples)[..., 1]
        lhs_samples = du_dt - self.k * d2u_dx2

        tx_ib = tf.concat([tx_init, tx_bnd], axis=0)
        u_ib = self.backbone(tx_ib, training=training)
        u_initial = u_ib[:tx_init.shape[0]]
        u_bnd = u_ib[tx_init.shape[0]:]

        return u_samples, lhs_samples, u_initial, u_bnd

    @tf.function
    def train_step(self, data):
        """
        Performs a training step.

        Args:
            data: The data to train on. First input is the samples, second input is the initial, \
                and third input is the boundary data. First output is the exact solution for the samples, \
                second output is the exact rhs for the samples, third output is the exact solution for the initial, \
                and fourth output is the exact solution for the boundary.
        """
        
        inputs, outputs = data
        u_samples_exact, rhs_samples_exact, u_initial_exact, u_bnd_exact = outputs

        with tf.GradientTape() as tape:
            u_samples, lhs_samples, u_initial, u_bnd = self(inputs, training=True)

            loss_residual = tf.losses.mean_squared_error(rhs_samples_exact, lhs_samples)
            loss_initial = tf.losses.mean_squared_error(u_initial_exact, u_initial)
            loss_boundary = tf.losses.mean_squared_error(u_bnd_exact, u_bnd)
            loss = self._loss_residual_weight * loss_residual + self._loss_initial_weight * loss_initial + \
                self._loss_boundary_weight * loss_boundary

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.mae_tracker.update_state(u_samples_exact, u_samples)
        self.loss_residual_tracker.update_state(loss_residual)
        self.loss_initial_tracker.update_state(loss_initial)
        self.loss_boundary_tracker.update_state(loss_boundary)

        return {m.name: m.result() for m in self.metrics}

    @property
    def metrics(self):
        """
        Returns the metrics of the model.
        """
        return [self.loss_residual_tracker, self.loss_initial_tracker, self.loss_boundary_tracker, self.mae_tracker]


class WavePinn(tf.keras.Model):
    """
    A model that solves the wave equation.
    """

    def __init__(self, backbone: "tf.keras.Model", c: float, loss_residual_weight=1.0, loss_initial_weight=1.0, \
        loss_boundary_weight=1.0, **kwargs):
        """
        Initializes the model.

        Args:
            backbone: The backbone model.
            c: The wave speed.
            loss_residual_weight: The weight of the residual loss.
            loss_initial_weight: The weight of the initial loss.
            loss_boundary_weight: The weight of the boundary loss.
        """
        super().__init__(**kwargs)
        self.backbone = backbone
        self.c = c
        self.loss_residual_tracker = tf.keras.metrics.Mean(name=LOSS_RESIDUAL)
        self.loss_initial_tracker = tf.keras.metrics.Mean(name=LOSS_INITIAL)
        self.loss_boundary_tracker = tf.keras.metrics.Mean(name=LOSS_BOUNDARY)
        self.mae_tracker = tf.keras.metrics.MeanAbsoluteError(name=MEAN_ABSOLUTE_ERROR)
        self._loss_residual_weight = tf.Variable(loss_residual_weight, trainable=False, name="loss_residual_weight", dtype=tf.keras.backend.floatx())
        self._loss_boundary_weight = tf.Variable(loss_boundary_weight, trainable=False, name="loss_boundary_weight", dtype=tf.keras.backend.floatx())
        self._loss_initial_weight = tf.Variable(loss_initial_weight, trainable=False, name="loss_initial_weight", dtype=tf.keras.backend.floatx())

    def set_loss_weights(self, loss_residual_weight: float, loss_initial_weight: float, loss_boundary_weight: float):
        """
        Sets the loss weights.

        Args:
            loss_residual_weight: The weight of the residual loss.
            loss_initial_weight: The weight of the initial loss.
            loss_boundary_weight: The weight of the boundary loss.
        """
        self._loss_residual_weight.assign(loss_residual_weight)
        self._loss_boundary_weight.assign(loss_boundary_weight)
        self._loss_initial_weight.assign(loss_initial_weight)

    @tf.function
    def call(self, inputs, training=False):
        """
        Performs a forward pass.

        Args:
            inputs: The inputs to the model. First input is the samples, second input is the initial, \
                and third input is the boundary data.
            training: Whether or not the model is training.

        Returns:
            The solution for the residual samples, the lhs residual, the solution for the initial samples, \
                and the solution for the boundary samples.
        """
        
        tx_samples = inputs[0]
        tx_init = inputs[1]
        tx_bnd = inputs[2]

        with tf.GradientTape(watch_accessed_variables=False) as tape2:
            tape2.watch(tx_samples)
            
            with tf.GradientTape(watch_accessed_variables=False) as tape1:
                tape1.watch(tx_samples)
                u_samples = self.backbone(tx_samples, training=training)

            first_order = tape1.batch_jacobian(u_samples, tx_samples)
        second_order = tape2.batch_jacobian(first_order, tx_samples)
        d2u_dt2 = second_order[..., 0, 0]
        d2u_dx2 = second_order[..., 1, 1]
        lhs_samples = d2u_dt2 - (self.c ** 2) * d2u_dx2

        u_bnd = self.backbone(tx_bnd, training=training)

        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(tx_init)
            u_initial = self.backbone(tx_init, training=training)
        du_dt_init = tape.batch_jacobian(u_initial, tx_init)[..., 0]

        return u_samples, lhs_samples, u_initial, du_dt_init, u_bnd

    @tf.function
    def train_step(self, data):
        """
        Performs a training step.

        Args:
            data: The data to train on. First input is the samples, second input is the initial, \
                and third input is the boundary data. The outputs are the exact solutions for the samples, \
                the exact rhs for the samples, the exact solution for the initial, the exact derivative for the initial, \
                and the exact solution for the boundary.
        """

        inputs, outputs = data
        u_samples_exact, rhs_samples_exact, u_initial_exact, du_dt_init_exact, u_bnd_exact = outputs

        with tf.GradientTape() as tape:
            u_samples, lhs_samples, u_initial, du_dt_init, u_bnd = self(inputs, training=True)

            loss_residual = tf.losses.mean_squared_error(rhs_samples_exact, lhs_samples)
            loss_initial_neumann = tf.losses.mean_squared_error(du_dt_init_exact, du_dt_init)
            loss_initial_dirichlet = tf.losses.mean_squared_error(u_initial_exact, u_initial)
            loss_initial = loss_initial_neumann + loss_initial_dirichlet
            loss_boundary = tf.losses.mean_squared_error(u_bnd_exact, u_bnd)
            loss = self._loss_residual_weight * loss_residual + self._loss_initial_weight * loss_initial + \
                self._loss_boundary_weight * loss_boundary

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.mae_tracker.update_state(u_samples_exact, u_samples)
        self.loss_residual_tracker.update_state(loss_residual)
        self.loss_initial_tracker.update_state(loss_initial)
        self.loss_boundary_tracker.update_state(loss_boundary)

        return {m.name: m.result() for m in self.metrics}

    @property
    def metrics(self):
        """
        Returns the metrics of the model.
        """
        return [self.loss_residual_tracker, self.loss_initial_tracker, self.loss_boundary_tracker, self.mae_tracker]


"""
Utility functions for saving and loading models, plots, logs, etc.
"""

import os
from pathlib import Path
from modules.models import LOSS_BOUNDARY, LOSS_INITIAL, LOSS_RESIDUAL, MEAN_ABSOLUTE_ERROR
import numpy as np
import pandas as pd
import scipy.io
import tensorflow as tf

def get_id(layers, optimizer, initialization, activation):
    """
    Generates a unique id for a model based on its layers, optimizer, initialization, and activation.
    """
    return str(layers)+ "," + type(optimizer).__name__ + "," + type(initialization).__name__ + "," + activation.__name__

def save_model(model, id, directory):
    """
    Saves a model to directory/models/id.h5
    """

    model_directory = os.path.join(directory, "models")
    Path(model_directory).mkdir(parents=True, exist_ok=True)
    name = id + ".h5"
    path = os.path.join(model_directory, name)
    model.save(path)

def save_history_csv(history, id, directory):
    """
    saves the history of a model to directory/history/id.csv
    """

    log_directory = os.path.join(directory, "logs")
    Path(log_directory).mkdir(parents=True, exist_ok=True)
    name = id+".csv"
    path = os.path.join(log_directory, name)
    resid_loss = history[LOSS_RESIDUAL]
    df = pd.DataFrame(np.array(resid_loss), columns = [LOSS_RESIDUAL])
    if LOSS_INITIAL in history and len(history[LOSS_INITIAL]) > 0:
        df[LOSS_INITIAL] = np.array(history[LOSS_INITIAL])
    if LOSS_BOUNDARY in history and len(history[LOSS_BOUNDARY]) > 0:
        df[LOSS_BOUNDARY] = np.array(history[LOSS_BOUNDARY])
    if MEAN_ABSOLUTE_ERROR in history and len(history[MEAN_ABSOLUTE_ERROR]) > 0:
        df[MEAN_ABSOLUTE_ERROR] = np.array(history[MEAN_ABSOLUTE_ERROR])
    df.to_csv(path)

def get_train_plot_name(id, directory):
    """
    Returns the path to the training plot for a model with id in directory/plots
    """
    
    plot_directory = os.path.join(directory, "plots")
    Path(plot_directory).mkdir(parents=True, exist_ok=True)
    name = id+'.jpg'
    return os.path.join(plot_directory, name)


def load_mat_data(path):
    """
    Loads the mat data from path into a dictionary
    """
    return scipy.io.loadmat(path)

class PrintLossCallback(tf.keras.callbacks.Callback):
    """
    Callback for printing the loss each n epochs
    """
    
    def __init__(self, n):
        super().__init__()
        self.n = n

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.n == 0:
            print("Epoch: ", epoch, "Lr: ", f"{logs[LOSS_RESIDUAL]:.5f}", "Li: ", \
                f"{logs[LOSS_INITIAL]:.5f}", "Lb: ", f"{logs[LOSS_BOUNDARY]:.5f}", "MAE: ", \
                    f"{logs[MEAN_ABSOLUTE_ERROR]:.5f}")
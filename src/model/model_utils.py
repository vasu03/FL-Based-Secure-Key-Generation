"""
- File:     src/model/model_utils.py
- Desc:     Utilities to create a compact Keras model and helpers to get/set weights
- Author:   Vasu Makadia
- License:  Apache License 2.0
"""


# Import required modules
import os
import warnings
from typing import List
import numpy as np

# To supress tensorflow warnings related to CUDA drivers
# "0" - all, "1" - info, "2" - warning, "3" = error
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISISBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")

import tensorflow as tf

def create_key_model (input_dim: int = 16, output_dim: int = 16) -> tf.keras.Model:
    """
    Brief:
        Build a small fully-connected Keras model which will be used as key-generator function
    Parameters:
        input_dim (int):    dimensionality of input vector (challenge + optional features)
        output_dim (int):   dimensionality of output vector used to derive keys
    Returns:
        tf.keras.Model:     a compiled Keras model ready for training and predictions
    """
    # Input layer with specified dimesions
    inp = tf.keras.Input(shape=(input_dim,), name="challenge_input")
    # Densly connected first hidden layer with ReLU activation
    x = tf.keras.layers.Dense(32, activation="relu", name="hidden_fc1")(inp)
    # Densly connected second hidden layer with ReLU activation
    x = tf.keras.layers.Dense(32, activation="relu", name="hidden_fc2")(x)
    # Output layer (linear -> will be quantized later)
    out = tf.keras.layers.Dense(output_dim, activation=None, name="output_vec")(x)

    # Create a model instance
    model = tf.keras.Model(inputs=inp, outputs=out, name="keygen_model")
    # Compile the model with MSE regression loss and Stochastic Gradient Descent for local training
    model.compile(
        optimizer = tf.keras.optimizers.SGD(learning_rate = 0.05),
        loss = "mse"
    )

    return model


def get_weights_as_numpy (model: tf.keras.Model) -> List[np.ndarray]:
    """
    Brief:
        Extract model weights as a list of numpy arrays (convenient for serialization)
    Parameters:
        model (tf.keras.Model):     A compiled Keras Model
    Returns:
        List[np.ndarray]:           List of numpy arrays corresponding to model weights
    """
    # Get the weights of the model as a list of numpy arrays
    return model.get_weights()


def set_weights_from_numpy (model: tf.keras.Model, weights: List[np.ndarray]) -> None:
    """
    Brief:
        Set model weights from a list of numpy arrays
    Parameters:
        model (tf.keras.Model):     A compiled Keras model
        weights (List[np.ndarray]): Weights to be set on the model
    Returns:
        None
    """
    # Set the weights on the model
    model.set_weights(weights)


def average_weights (weight_lists: List[List[np.ndarray]]) -> List[np.ndarray]:
    """
    Brief:
        Compute element-wise average of multiple model-weight lists (Federated Averaging)
    Parameters:
        weight_lists (List[List[np.ndarray]]):  List of weight lists from clients
    Returns:
        List[np.ndarray]:                       Averaged weights
    """
    # No. of clients that contributed
    num_clients = len(weight_lists)
    # Initialize accumulator with zero shaped like first client's weights
    avg_weights = []

    for layer_idx in range (len(weight_lists[0])):
        # sum up all clients' layer weights
        weight_sum = sum([client_weights[layer_idx] for client_weights in weight_lists])
        # Average by number of clients
        avg_layer = weight_sum / float(num_clients)
        avg_weights.append(avg_layer)

    return avg_weights


def print_model_weights (weights: List[np.ndarray], prefix: str = "") -> None:
    """
    Brief:
        To print the model weights for debugging purpose
    Parameters:
        weights (List[np.ndarray]):     A list of model weights
        prefix (str):                   optional prefix string for logging
    Returns:
        None
    """
    print(f"\n[{prefix}] Model Weights:")
    for i, w in enumerate(weights):
        print(f"Layer {i}: shape={w.shape}, values={w.flatten()[:5]}")

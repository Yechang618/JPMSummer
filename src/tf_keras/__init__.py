"""Compatibility shim for tf_keras imports.

This module re-exports tensorflow.keras to provide compatibility with code
that expects tf_keras to be available. The structure mirrors TensorFlow's
organization to support internal imports.
"""
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from tensorflow import keras

# Core Keras modules
layers = keras.layers
models = keras.models
optimizers = keras.optimizers
losses = keras.losses
metrics = keras.metrics
callbacks = keras.callbacks
utils = keras.utils
preprocessing = keras.preprocessing

# Required internal modules
backend = keras.backend
__internal__ = keras.__internal__
__version__ = keras.__version__

# Module exports
__all__ = [
    "layers",
    "models", 
    "optimizers",
    "losses",
    "metrics",
    "callbacks",
    "utils",
    "preprocessing",
    "backend",
    "__internal__",
    "__version__",
]
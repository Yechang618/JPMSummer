"""Compatibility shim for tf_keras imports.

This module re-exports tensorflow.keras to provide compatibility with code
that expects tf_keras to be available. The structure mirrors TensorFlow's
organization to support internal imports.

Usage:
    import tf_keras
    model = tf_keras.models.Sequential()
    layer = tf_keras.layers.Dense(10)
"""
import os
import sys
from importlib.util import find_spec

# Disable oneDNN optimization warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Check if tensorflow is available
if find_spec("tensorflow") is None:
    raise ImportError(
        "tensorflow is required but not installed. "
        "Please install with: pip install tensorflow"
    )

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
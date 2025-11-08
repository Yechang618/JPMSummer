"""Internal utilities re-exported from tensorflow.keras.

This module provides internal utilities needed by some TensorFlow components.
It should not be imported directly by user code.
"""
from tensorflow import keras

__internal__ = keras.__internal__
__all__ = ["__internal__"]
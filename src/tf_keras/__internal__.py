"""Internal utilities re-exported from tensorflow.keras."""
import os
os.environ["KERAS_BACKEND"] = "tensorflow" # or "jax", "pytorch"

from tensorflow.keras import __internal__

__all__ = ["__internal__"]
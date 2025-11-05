"""Compatibility shim for `tf_keras` imports.

Some packages expect a top-level module named `tf_keras`. This project uses
`tensorflow` and `tensorflow.keras`. To avoid ModuleNotFoundError on
`import tf_keras`, provide a small shim that re-exports the commonly used
submodules from `tensorflow.keras`.

This file lives at the project root so that running scripts from the project
root (e.g. notebooks, `python -m test`) will find it on sys.path.
"""
from __future__ import annotations

try:
    import tensorflow as _tf
except Exception:  # pragma: no cover - environment dependent
    raise

# Re-export keras and commonly used submodules under the tf_keras name.
keras = _tf.keras
layers = keras.layers
models = keras.models
optimizers = keras.optimizers
losses = keras.losses
callbacks = keras.callbacks
metrics = keras.metrics
utils = keras.utils
preprocessing = keras.preprocessing

__all__ = [
    "keras",
    "layers",
    "models",
    "optimizers",
    "losses",
    "callbacks",
    "metrics",
    "utils",
    "preprocessing",
]

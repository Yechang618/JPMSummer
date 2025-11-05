"""Compatibility shim for `tf_keras` imports included in the package.

Some third-party code imports `tf_keras` as a top-level module. This shim
re-exports `tensorflow.keras` submodules so that `import tf_keras` and
`from tf_keras import layers` behave as expected after the package is
installed (e.g. `pip install -e .`).

This file is installed as part of the package so editable installs or PyPI
installs will make `tf_keras` available.
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

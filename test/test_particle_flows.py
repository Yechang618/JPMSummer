import unittest
import numpy as np
import tensorflow as tf

import warnings, os, sys
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore", category=DeprecationWarning)


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from models.particle_filters import (
    algorithm1_DH_filter,
    algorithm2_modified_DH_filter
)

@tf.function
def gamma_linear(x):
    """Observation: y = x_0 (first state component). Handles 1D and 2D inputs."""
    x = tf.convert_to_tensor(x)
    if x.shape.ndims == 1:
        # x is (dx,) → return scalar output as (1,)
        return tf.expand_dims(x[0], axis=0)  # shape: (1,)
    else:
        # x is (N, dx) → return (N, 1)
        return x[:, :1]
@tf.function
def dgamma_dx_linear_global(x):
    """Jacobian for global DH (single particle): dy/dx = [1, 0, ..., 0]"""
    x = tf.convert_to_tensor(x)
    dx = tf.shape(x)[0]  # x is 1D: (dx,)
    # Ensure consistent dtype
    ones = tf.constant([1.0], dtype=x.dtype)          # ← Critical fix
    zeros = tf.zeros(dx - 1, dtype=x.dtype)
    jac_row = tf.concat([ones, zeros], axis=0)        # (dx,)
    return tf.expand_dims(jac_row, axis=0)            # (1, dx)

@tf.function
def dgamma_dx_linear_batch(x):
    """Jacobian for local DH (handles 1D or 2D inputs)."""
    x = tf.convert_to_tensor(x)
    if x.shape.ndims == 1:
        # Single particle: x is (dx,) → return (1, dx)
        dx = tf.shape(x)[0]
        ones = tf.constant([1.0], dtype=x.dtype)
        zeros = tf.zeros(dx - 1, dtype=x.dtype)
        jac_row = tf.concat([ones, zeros], axis=0)    # (dx,)
        return tf.expand_dims(jac_row, axis=0)        # (1, dx)
    else:
        # Batch: x is (N, dx) → return (N, 1, dx)
        N = tf.shape(x)[0]
        dx = tf.shape(x)[1]
        ones = tf.ones((N, 1), dtype=x.dtype)         # (N, 1)
        zeros = tf.zeros((N, dx - 1), dtype=x.dtype)  # (N, dx-1)
        jac_rows = tf.concat([ones, zeros], axis=1)   # (N, dx)
        return tf.expand_dims(jac_rows, axis=1)       # (N, 1, dx)
    

class TestParticleFlows(unittest.TestCase):

    def test_algorithm1_DH_filter(self):
        T = 5
        y_seq = np.random.randn(T, 1).astype(np.float64)
        est = algorithm1_DH_filter(
            y_seq=y_seq,
            T=T,
            N=100,
            Psi=np.eye(2, dtype=np.float64),
            Q=np.eye(2, dtype=np.float64) * 0.1,
            R=np.eye(1, dtype=np.float64) * 0.2,
            gamma=gamma_linear,
            dgamma_dx=dgamma_dx_linear_global,  # ← updated
            x0_mean=np.zeros(2, dtype=np.float64),
            x0_cov=np.eye(2, dtype=np.float64),
            n_lambda=5,
            dtype=tf.float64
        )
        self.assertEqual(est.shape, (T, 2))
        self.assertFalse(np.any(np.isnan(est)))

    def test_algorithm2_modified_DH_filter(self):
        T = 5
        y_seq = np.random.randn(T, 1).astype(np.float64)
        est = algorithm2_modified_DH_filter(
            y_seq=y_seq,
            T=T,
            N=100,
            Psi=np.eye(2, dtype=np.float64),
            Q=np.eye(2, dtype=np.float64) * 0.1,
            R=np.eye(1, dtype=np.float64) * 0.2,
            gamma=gamma_linear,
            dgamma_dx=dgamma_dx_linear_batch,  # ← updated
            x0_mean=np.zeros(2, dtype=np.float64),
            x0_cov=np.eye(2, dtype=np.float64),
            n_lambda=5,
            dtype=tf.float64
        )
        self.assertEqual(est.shape, (T, 2))
        self.assertFalse(np.any(np.isnan(est)))
        
if __name__ == '__main__':
    unittest.main()
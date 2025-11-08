import numpy as np
import tensorflow as tf

from src.models import ExtendedKalmanFilter as ekf_mod


def test_ekf_basic_shapes_and_finite():
    # Simulate small nonlinear system (NumPy arrays)
    true_x, observations, Q, R = ekf_mod._simulate_simple_nonlinear(T=40, seed=2)

    # Define transition/observation functions using TensorFlow ops
    def f(x: tf.Tensor) -> tf.Tensor:
        return x + 0.05 * tf.square(x)

    def h(x: tf.Tensor) -> tf.Tensor:
        return x + 0.1 * tf.square(x)

    ekf = ekf_mod.ExtendedKalmanFilter(
        f=f,
        h=h,
        Q=tf.constant([[Q]], dtype=tf.float64),
        R=tf.constant([[R]], dtype=tf.float64),
        initial_mean=tf.constant([0.0], dtype=tf.float64),
        initial_cov=tf.constant([[1.0]], dtype=tf.float64),
    )

    means_tf, covs_tf, ll_tf = ekf.filter(observations)

    means = means_tf.numpy()
    covs = covs_tf.numpy()
    ll = float(ll_tf.numpy()) if hasattr(ll_tf, "numpy") else float(ll_tf)

    # Basic assertions: correct shapes
    assert means.shape == (observations.shape[0], 1)
    assert covs.shape == (observations.shape[0], 1, 1)

    # Finite entries
    assert np.isfinite(ll)
    assert np.all(np.isfinite(means))
    assert np.all(np.isfinite(covs))

    # Reasonable accuracy: RMSE less than a large threshold (sanity)
    rmse = np.sqrt(np.mean((means.squeeze() - true_x) ** 2))
    assert rmse < 5.0

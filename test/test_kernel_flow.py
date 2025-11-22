import numpy as np
import tensorflow as tf

from src.models.KernelFlow import KernelScalarFlow, KernelMatrixFlow


def test_kernel_scalar_flow_smoke():
    tf.random.set_seed(1)
    N = 20
    d = 5
    particles = tf.convert_to_tensor(np.random.randn(N, d), dtype=tf.float64)
    y = np.random.randn(d)
    H = np.eye(d)
    R = np.eye(d)

    flow = KernelScalarFlow(n_flow_steps=2, alpha=1.0 / N)
    x_new, logabs = flow.flow(particles, y, H, R)

    assert x_new.shape == (N, d)
    # logabs is scalar in this implementation
    assert tf.size(logabs).numpy() >= 1
    assert np.all(np.isfinite(x_new.numpy()))


def test_kernel_matrix_flow_smoke():
    tf.random.set_seed(2)
    N = 16
    d = 4
    particles = tf.convert_to_tensor(np.random.randn(N, d), dtype=tf.float64)
    y = np.random.randn(d)
    H = np.eye(d)
    R = np.eye(d)

    flow = KernelMatrixFlow(n_flow_steps=2, alpha=1.0 / N)
    x_new, logabs = flow.flow(particles, y, H, R)

    assert x_new.shape == (N, d)
    # logabs is per-particle vector
    assert tf.shape(logabs)[0].numpy() == N
    assert np.all(np.isfinite(x_new.numpy()))

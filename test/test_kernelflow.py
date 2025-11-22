import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from models.KernelFlow import KernelScalarFlow, KernelMatrixFlow


def test_kernelflows_basic():
    tf.random.set_seed(0)
    tfd = tfp.distributions
    N = 50
    d = 3

    prior = tfd.MultivariateNormalDiag(loc=tf.zeros(d, dtype=tf.float64), scale_diag=tf.ones(d, dtype=tf.float64))
    particles = prior.sample(N)

    H = tf.constant(np.eye(d), dtype=tf.float64)
    R = tf.constant(np.eye(d) * 0.3, dtype=tf.float64)

    ks = KernelScalarFlow(n_flow_steps=4)
    km = KernelMatrixFlow(n_flow_steps=4)

    # run scalar kernel flow
    x_s, log_s = ks.flow(particles, tf.zeros(d, dtype=tf.float64), H, R)
    # run matrix kernel flow
    x_m, log_m = km.flow(particles, tf.zeros(d, dtype=tf.float64), H, R)

    # basic sanity checks: shapes and finite values
    assert x_s.shape == (N, d)
    assert x_m.shape == (N, d)
    assert np.all(np.isfinite(x_s.numpy()))
    assert np.all(np.isfinite(x_m.numpy()))

    # logabs shapes: scalar flow returns rank-1 or scalar, matrix returns per-particle
    assert log_m.shape[0] == N

    # flows should move particles at least a little (not identical)
    assert not np.allclose(x_s.numpy(), particles.numpy())
    assert not np.allclose(x_m.numpy(), particles.numpy())

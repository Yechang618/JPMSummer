import numpy as np
import tensorflow as tf
from models.EDH import EDH, LEDH


def make_gaussian_particles(N, d, mean=None, cov=None, seed=1):
    rng = np.random.default_rng(seed)
    if mean is None:
        mean = np.zeros(d)
    if cov is None:
        cov = np.eye(d)
    samples = rng.multivariate_normal(mean, cov, size=N)
    return tf.convert_to_tensor(samples, dtype=tf.float64)


def test_edh_ledh_linear_equivalence():
    tf.random.set_seed(1)
    N = 200
    d = 3
    particles = make_gaussian_particles(N, d, seed=2)

    # Linear observation H and R
    H = tf.constant(np.eye(d), dtype=tf.float64)
    R = tf.constant(np.eye(d) * 0.5, dtype=tf.float64)

    # observation y
    true_x = np.arange(1, d + 1).astype(float)
    y = tf.convert_to_tensor(true_x, dtype=tf.float64)

    # Build an observation function h(x)=H x and pass R into the filters
    def make_h(H):
        return lambda x: tf.reshape(tf.matmul(H, tf.reshape(x, [-1, 1])), [-1])

    h_fn = make_h(H)

    # EDH (global linearization at mean) and LEDH (per-particle) should match for linear H
    edh = EDH(num_particles=N, f=None, h=h_fn, state_dim=d, observation_dim=d, R=R, dtype=tf.float64)
    ledh = LEDH(num_particles=N, f=None, h=h_fn, state_dim=d, observation_dim=d, R=R, dtype=tf.float64)

    edh.initialize(particles=particles)
    ledh.initialize(particles=particles)

    edh.n_flow_steps = 5
    ledh.n_flow_steps = 5

    # Run EDH and LEDH; LEDH.update returns (state_estimate, P)
    edh.update(y=y)
    ledh_m, ledh_P = ledh.update(y=y)

    # EDH stores the analytic posterior in .m and .P; LEDH returns (x_hat, P)
    edh_m = edh.m.numpy()
    edh_P = edh.P.numpy()

    # Check EDH and LEDH analytic outputs are close (relaxed tolerance to allow integration/jitter differences)
    assert np.allclose(edh_m, ledh_m.numpy(), atol=1e-1, rtol=1e-6)
    assert np.allclose(edh_P, ledh_P.numpy(), atol=1e-1, rtol=1e-6)


def test_edh_ledh_small_1d():
    tf.random.set_seed(3)
    N = 100
    d = 1
    particles = make_gaussian_particles(N, d, seed=4)
    H = tf.constant([[1.0]], dtype=tf.float64)
    R = tf.constant([[0.2]], dtype=tf.float64)
    y = tf.convert_to_tensor([0.5], dtype=tf.float64)

    def make_h(H):
        return lambda x: tf.reshape(tf.matmul(H, tf.reshape(x, [-1, 1])), [-1])

    h_fn = make_h(H)

    edh = EDH(num_particles=N, f=None, h=h_fn, state_dim=d, observation_dim=d, R=R, dtype=tf.float64)
    ledh = LEDH(num_particles=N, f=None, h=h_fn, state_dim=d, observation_dim=d, R=R, dtype=tf.float64)

    edh.initialize(particles=particles)
    ledh.initialize(particles=particles)

    edh.n_flow_steps = 4
    ledh.n_flow_steps = 4

    edh.update(y=y)
    ledh_m, ledh_P = ledh.update(y=y)

    edh_m = edh.m.numpy()
    edh_P = edh.P.numpy()

    assert np.allclose(edh_m, ledh_m.numpy(), atol=1e-1, rtol=1e-6)
    assert np.allclose(edh_P, ledh_P.numpy(), atol=1e-1, rtol=1e-6)

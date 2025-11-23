import numpy as np
import tensorflow as tf

from src.data.data import StochasticVariationalData


def test_sample_shapes_and_types():
    T = 5
    n_state = 2
    sv = StochasticVariationalData(alpha=0.9, sigma=0.5, beta=0.7, n_state=n_state, dtype=tf.float64)
    x, y = sv.sample(num_steps=T, seed=123)

    # x should be (T+1, n_state), y should be (T, n_state)
    assert isinstance(x, tf.Tensor)
    assert isinstance(y, tf.Tensor)
    x_np = x.numpy()
    y_np = y.numpy()
    assert x_np.shape == (T + 1, n_state)
    assert y_np.shape == (T, n_state)


def test_seed_reproducibility_and_variation():
    T = 6
    n_state = 3
    sv = StochasticVariationalData(alpha=0.8, sigma=0.3, beta=0.4, n_state=n_state, dtype=tf.float64)
    # Reset global RNG state to make calls deterministic across invocations
    tf.random.set_seed(0)
    np.random.seed(0)
    x1, y1 = sv.sample(num_steps=T, seed=123)

    tf.random.set_seed(0)
    np.random.seed(0)
    x2, y2 = sv.sample(num_steps=T, seed=123)

    tf.random.set_seed(0)
    np.random.seed(0)
    x3, y3 = sv.sample(num_steps=T, seed=456)

    # Same seed -> identical samples
    np.testing.assert_allclose(x1.numpy(), x2.numpy())
    np.testing.assert_allclose(y1.numpy(), y2.numpy())

    # Different seed -> different samples (very high probability)
    assert not (np.allclose(x1.numpy(), x3.numpy()) and np.allclose(y1.numpy(), y3.numpy()))

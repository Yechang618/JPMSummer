import unittest
import numpy as np
import tensorflow as tf

import warnings, os, sys
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore", category=DeprecationWarning)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from data.data import (
    LinearGaussianSSM,
    StochasticVolatilityModel,
    RangeBearingSSM,
    Lorenz96NonGaussianSSM
)

class TestDataGeneration(unittest.TestCase):

    def test_linear_gaussian_ssm(self):
        ssm = LinearGaussianSSM(
            transition_matrix=[[0.9, 0.1], [0.0, 0.8]],
            observation_matrix=[[1.0, 0.0]],
            transition_cov=[[0.1, 0.0], [0.0, 0.1]],
            observation_cov=[[0.2]],
            initial_mean=[0.0, 0.0],
            initial_cov=[[1.0, 0.0], [0.0, 1.0]],
            dtype=tf.float64
        )
        x, y = ssm.sample(10, seed=42)
        self.assertEqual(x.shape, (11, 2))
        self.assertEqual(y.shape, (10, 1))
        self.assertEqual(x.dtype, tf.float64)
        self.assertEqual(y.dtype, tf.float64)

    def test_stochastic_volatility_model(self):
        ssm = StochasticVolatilityModel(
            alpha=0.95, sigma=0.2, beta=0.5, n_state=1, dtype=tf.float64
        )
        x, y = ssm.sample(10, seed=42)
        self.assertEqual(x.shape, (11, 1))
        self.assertEqual(y.shape, (10, 1))
        self.assertTrue(tf.reduce_all(y != 0))  # non-zero observations

    def test_range_bearing_ssm(self):
        ssm = RangeBearingSSM(dtype=tf.float64)
        x, y = ssm.sample(10, seed=42)
        self.assertEqual(x.shape, (11, 4))
        self.assertEqual(y.shape, (10, 2))
        r, b = y[:, 0], y[:, 1]
        self.assertTrue(tf.reduce_all(r >= 0))          # range ≥ 0
        self.assertTrue(tf.reduce_all(b >= -np.pi))     # bearing in [-π, π]
        self.assertTrue(tf.reduce_all(b <= np.pi))

    def test_lorenz96_ssm(self):
        ssm = Lorenz96NonGaussianSSM(dim=5, seed=42)
        x, y = ssm.generate_data(T=10)
        self.assertEqual(x.shape, (10, 5))
        self.assertEqual(y.shape, (10, 5))
        self.assertIsInstance(x, np.ndarray)
        self.assertIsInstance(y, np.ndarray)

if __name__ == '__main__':
    unittest.main()
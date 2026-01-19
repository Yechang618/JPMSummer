import unittest
import numpy as np
import tensorflow as tf

import warnings, os, sys
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore", category=DeprecationWarning)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))


from models.kalman_filters import (
    KalmanFilter,
    ExtendedKalmanFilter,
    UnscentedKalmanFilter
)

class TestKalmanFilters(unittest.TestCase):

    def test_kalman_filter(self):
        kf = KalmanFilter(
            transition_matrix=[[0.9]],
            observation_matrix=[[1.0]],
            transition_cov=[[0.1]],
            observation_cov=[[0.2]],
            initial_mean=[0.0],
            initial_cov=[[1.0]],
            dtype=tf.float64
        )
        obs = tf.constant([[0.1], [0.2], [0.3]], dtype=tf.float64)
        means, covs, ll = kf.filter(obs)
        self.assertEqual(means.shape, (3, 1))
        self.assertEqual(covs.shape, (3, 1, 1))
        self.assertTrue(tf.reduce_all(tf.linalg.diag_part(covs) > 0))
        self.assertFalse(tf.math.is_nan(ll))

    def test_extended_kalman_filter(self):
        def f(x): return x
        def h(x): return x ** 2
        ekf = ExtendedKalmanFilter(
            f=f, h=h,
            Q=[[0.1]], R=[[0.2]],
            initial_mean=[0.0], initial_cov=[[1.0]],
            dtype=tf.float64
        )
        obs = tf.constant([[0.1], [0.4], [0.9]], dtype=tf.float64)
        means, covs, ll = ekf.filter(obs)
        self.assertEqual(means.shape, (3, 1))
        self.assertEqual(covs.shape, (3, 1, 1))

    def test_unscented_kalman_filter(self):
        def f(x): return x
        def h(x): return x ** 2
        ukf = UnscentedKalmanFilter(
            f=f, h=h,
            Q=[[0.1]], R=[[0.2]],
            initial_mean=np.array([0.0]),
            initial_cov=np.array([[1.0]]),
            dtype=tf.float64
        )
        obs = np.array([[0.1], [0.4], [0.9]], dtype=np.float64)
        means, covs, ll = ukf.filter(obs)
        self.assertEqual(means.shape, (3, 1))
        self.assertEqual(covs.shape, (3, 1, 1))

if __name__ == '__main__':
    unittest.main()
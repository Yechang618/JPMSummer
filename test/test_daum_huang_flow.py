
import os
import sys
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
# Ensure src is on the path for import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from models.DaumHuangFlow import DaumHuangFlow

tfd = tfp.distributions

def test_daum_huang_linear_gaussian():
    # 1D linear-Gaussian test
    N = 1000
    d = 1
    prior = tfd.MultivariateNormalDiag(loc=tf.zeros(d, dtype=tf.float64), scale_diag=tf.ones(d, dtype=tf.float64))
    dhf = DaumHuangFlow(num_particles=N, state_dim=d, dtype=tf.float64)
    dhf.initialize(initial_dist=prior)

    # Observation model y = H x + noise
    H = tf.constant([[1.0]], dtype=tf.float64)
    R = tf.constant([[0.5]], dtype=tf.float64)
    true_x = tf.constant([2.0], dtype=tf.float64)
    y = tf.reshape(true_x + tf.random.normal([1], stddev=np.sqrt(R.numpy()[0,0]), dtype=tf.float64), [-1])

    dhf.n_flow_steps = 10
    dhf.update(y=y, H=H, R=R)

    est = dhf.get_state_estimate().numpy()
    cov = dhf.get_state_covariance().numpy()

    # Analytic posterior
    P0 = np.array([[1.0]])
    P_post = np.linalg.inv(np.linalg.inv(P0) + H.numpy().T @ np.linalg.inv(R.numpy()) @ H.numpy())
    m_post = (P_post @ (np.linalg.inv(P0) @ np.zeros((d,)) + H.numpy().T @ np.linalg.inv(R.numpy()) @ y.numpy()))

    # Assert mean and covariance are close
    assert np.allclose(est, m_post, atol=0.05), f"Mean mismatch: {est} vs {m_post}"
    assert np.allclose(cov, P_post, atol=0.05), f"Covariance mismatch: {cov} vs {P_post}"

if __name__ == "__main__":
    test_daum_huang_linear_gaussian()
    print("DaumHuangFlow 1D linear-Gaussian test passed.")

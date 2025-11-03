"""Kalman filter implementation using TensorFlow and TensorFlow Probability.

This module provides a small, clear Kalman filter implementation based on
matrix equations. It uses `tensorflow` for tensor math and `tensorflow_probability`
for log-probability evaluation of the Gaussian innovation.

Class
------
- KalmanFilter: construct with system matrices and run `filter()` on observations.

Example
-------
>>> import numpy as np
>>> from src.models.KalmanFilter import KalmanFilter
>>> # small 1D example omitted here; run `python -m src.models.KalmanFilter` to see a demo
"""
from __future__ import annotations

from typing import Optional, Tuple

import tensorflow as tf
import tensorflow_probability as tfp

tfb = tfp.bijectors
tfd = tfp.distributions


class KalmanFilter:
    """A simple Kalman filter implemented using TensorFlow operations.

    This implements the discrete-time linear Gaussian Kalman filter
    (predict-update loop) for the model:

        x_t = A x_{t-1} + q_t,   q_t ~ N(0, Q)
        y_t = H x_t     + r_t,   r_t ~ N(0, R)

    Parameters
    ----------
    transition_matrix: array-like, shape (state_dim, state_dim)
    observation_matrix: array-like, shape (obs_dim, state_dim)
    transition_cov: array-like, shape (state_dim, state_dim)
    observation_cov: array-like, shape (obs_dim, obs_dim)
    initial_mean: array-like, shape (state_dim,)
    initial_cov: array-like, shape (state_dim, state_dim)
    dtype: tf.DType, default tf.float64
    """

    def __init__(
        self,
        transition_matrix,
        observation_matrix,
        transition_cov,
        observation_cov,
        initial_mean,
        initial_cov,
        dtype: tf.DType = tf.float64,
    ) -> None:
        self.dtype = dtype
        self.A = tf.convert_to_tensor(transition_matrix, dtype=self.dtype)
        self.H = tf.convert_to_tensor(observation_matrix, dtype=self.dtype)
        self.Q = tf.convert_to_tensor(transition_cov, dtype=self.dtype)
        self.R = tf.convert_to_tensor(observation_cov, dtype=self.dtype)
        self.m0 = tf.convert_to_tensor(initial_mean, dtype=self.dtype)
        self.P0 = tf.convert_to_tensor(initial_cov, dtype=self.dtype)

        # Dimensions check
        self.state_dim = int(self.A.shape[0])
        self.obs_dim = int(self.H.shape[0])

        assert self.A.shape[0] == self.A.shape[1], "transition_matrix must be square"
        assert self.Q.shape == (self.state_dim, self.state_dim)
        assert self.H.shape[1] == self.state_dim
        assert self.R.shape == (self.obs_dim, self.obs_dim)

    def filter(self, observations: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Run the Kalman filter on a sequence of observations.

        Parameters
        ----------
        observations: Tensor, shape [T, obs_dim]

        Returns
        -------
        filtered_means: Tensor, shape [T, state_dim]
        filtered_covs: Tensor, shape [T, state_dim, state_dim]
        loglik: Tensor scalar, total log-likelihood of observations
        """
        observations = tf.convert_to_tensor(observations, dtype=self.dtype)
        T = tf.shape(observations)[0]

        m = tf.identity(self.m0)
        P = tf.identity(self.P0)

        filtered_means = tf.TensorArray(self.dtype, size=T)
        filtered_covs = tf.TensorArray(self.dtype, size=T)

        total_loglik = tf.constant(0.0, dtype=self.dtype)

        I = tf.eye(self.state_dim, dtype=self.dtype)

        # Loop over time steps. We keep this simple and readable.
        for t in tf.range(T):
            y = observations[t]

            # Predict
            m_pred = tf.linalg.matvec(self.A, m)
            P_pred = self.A @ P @ tf.transpose(self.A) + self.Q

            # Innovation
            y_pred = tf.linalg.matvec(self.H, m_pred)
            S = self.H @ P_pred @ tf.transpose(self.H) + self.R

            # Numerically stable inverse using cholesky
            chol_S = tf.linalg.cholesky(S)
            # Kalman gain: K = P_pred H^T S^{-1}
            # Solve S * X = H * P_pred^T^T  => use triangular_solve
            # compute K = P_pred @ H^T @ S^{-1}
            K = tf.linalg.matmul(P_pred, tf.transpose(self.H))
            K = tf.linalg.cholesky_solve(chol_S, tf.transpose(K))
            K = tf.transpose(K)

            innovation = y - y_pred

            # Update
            m = m_pred + tf.linalg.matvec(K, innovation)
            KH = K @ self.H
            P = (I - KH) @ P_pred @ tf.transpose(I - KH) + KH @ self.R @ tf.transpose(KH)  # Joseph form

            # Log-likelihood contribution
            mvn = tfd.MultivariateNormalFullCovariance(loc=y_pred, covariance_matrix=S)
            total_loglik += mvn.log_prob(y)

            filtered_means = filtered_means.write(t, m)
            filtered_covs = filtered_covs.write(t, P)

        filtered_means = filtered_means.stack()
        filtered_covs = filtered_covs.stack()

        return filtered_means, filtered_covs, total_loglik


def _simulate_1d(T=50, seed: Optional[int] = 1):
    """Simulate a small 1D linear Gaussian system for demo and testing."""
    import numpy as np

    rng = np.random.default_rng(seed)
    A = np.array([[1.0]])
    H = np.array([[1.0]])
    Q = np.array([[0.1]])
    R = np.array([[0.5]])
    x = np.zeros((T + 1, 1))
    y = np.zeros((T, 1))
    x[0] = 0.0
    for t in range(1, T + 1):
        x[t] = A @ x[t - 1] + rng.normal(scale=np.sqrt(Q[0, 0]))
    for t in range(T):
        y[t] = H @ x[t + 1] + rng.normal(scale=np.sqrt(R[0, 0]))
    return A, H, Q, R, x[1:], y


def main():
    """Demonstration runner: simulate 1D series, filter it, and print results."""
    import numpy as np

    A, H, Q, R, true_states, observations = _simulate_1d(T=60)

    kf = KalmanFilter(
        transition_matrix=A,
        observation_matrix=H,
        transition_cov=Q,
        observation_cov=R,
        initial_mean=np.zeros((1,)),
        initial_cov=np.eye(1) * 1.0,
    )

    fm, Fc, ll = kf.filter(observations.astype(float))

    print("Log-likelihood:", float(ll))
    print("First 5 filtered means:")
    print(np.array(fm[:5]).squeeze())


if __name__ == "__main__":
    main()

"""Extended Kalman filter implemented with TensorFlow and TensorFlow Probability.

This implementation linearizes the nonlinear transition and observation
functions using automatic differentiation (tf.GradientTape). It returns
TensorFlow tensors (eager execution) and uses TFP for log-probability
calculations.
"""
from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfb = tfp.bijectors
tfd = tfp.distributions


class ExtendedKalmanFilter:
    """Extended Kalman Filter using TensorFlow operations.

    Parameters
    ----------
    f : callable
        Transition function f(x) -> x_next. Must accept and return tf.Tensor of shape (state_dim,).
    h : callable
        Observation function h(x) -> y. Must accept and return tf.Tensor of shape (obs_dim,).
    Q : array-like or tf.Tensor
        Process noise covariance (state_dim x state_dim)
    R : array-like or tf.Tensor
        Observation noise covariance (obs_dim x obs_dim)
    initial_mean : array-like or tf.Tensor
        Initial state mean (state_dim,)
    initial_cov : array-like or tf.Tensor
        Initial state covariance (state_dim x state_dim)
    dtype : tf.DType
        Floating dtype to use (default tf.float64)
    """

    def __init__(
        self,
        f: Callable[[tf.Tensor], tf.Tensor],
        h: Callable[[tf.Tensor], tf.Tensor],
        Q,
        R,
        initial_mean,
        initial_cov,
        dtype: tf.DType = tf.float64,
        verbose_or_not: bool = False,
    ) -> None:
        self.f = f
        self.h = h
        self.dtype = dtype

        self.Q = tf.convert_to_tensor(Q, dtype=self.dtype)
        self.R = tf.convert_to_tensor(R, dtype=self.dtype)

        self.m0 = tf.convert_to_tensor(initial_mean, dtype=self.dtype)
        self.P0 = tf.convert_to_tensor(initial_cov, dtype=self.dtype)
        self.verbose = bool(verbose_or_not)

        self.state_dim = int(self.P0.shape[0])
        self.obs_dim = int(self.R.shape[0])

        # Validate dimensions
        assert tuple(self.Q.shape) == (self.state_dim, self.state_dim)
        assert tuple(self.R.shape) == (self.obs_dim, self.obs_dim)
        
        # initialize running state for step() convenience (keeps filter stateful)
        self.m = tf.identity(self.m0)
        self.P = tf.identity(self.P0)
        # Evaluation histories
        # Per-step RMSE of state estimate (float list). Appended if true states provided.
        self.rmse_history = []
        # Per-step observation log-likelihood (float list)
        self.ll_history = []

    def reset(self) -> None:
        """Reset the internal filter state to the initial mean/covariance."""
        self.m = tf.identity(self.m0)
        self.P = tf.identity(self.P0)

    def step(self, observation, true_state=None) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Perform a single predict-update step and update internal state.

        Parameters
        ----------
        observation : array-like or tf.Tensor
            Observation for the current time step (obs_dim,) or scalar for 1D.

        Returns
        -------
        m : tf.Tensor
            Posterior mean after update (state_dim,)
        P : tf.Tensor
            Posterior covariance after update (state_dim, state_dim)
        loglik : tf.Tensor
            Log-likelihood contribution for this observation (scalar)
        """
        y = tf.convert_to_tensor(observation, dtype=self.dtype)
        # normalize shape
        if y.shape.rank == 1 and tf.shape(y)[0] != self.obs_dim:
            y = tf.reshape(y, (self.obs_dim,))
        elif y.shape.rank == 0:
            y = tf.reshape(y, (1,))

        I = tf.eye(self.state_dim, dtype=self.dtype)

        if self.verbose:
            print("ExtendedKalmanFilter: step() starting")

        # Predict
        m_pred = tf.convert_to_tensor(self.f(self.m), dtype=self.dtype)
        F = self._jacobian(self.f, self.m)
        P_pred = F @ self.P @ tf.transpose(F) + self.Q

        # Innovation
        y_pred = tf.convert_to_tensor(self.h(m_pred), dtype=self.dtype)
        H = self._jacobian(self.h, m_pred)
        S = H @ P_pred @ tf.transpose(H) + self.R

        # Kalman gain via Cholesky solve for numerical stability
        chol_S = tf.linalg.cholesky(S)
        K = tf.linalg.matmul(P_pred, tf.transpose(H))
        K = tf.linalg.cholesky_solve(chol_S, tf.transpose(K))
        K = tf.transpose(K)

        innovation = y - y_pred

        # Update
        self.m = m_pred + tf.linalg.matvec(K, innovation)
        KH = K @ H
        self.P = (I - KH) @ P_pred @ tf.transpose(I - KH) + KH @ self.R @ tf.transpose(KH)

        # Log-likelihood for this observation
        mvn = tfd.MultivariateNormalFullCovariance(loc=y_pred, covariance_matrix=S)
        loglik = mvn.log_prob(y)

        # Record per-step diagnostics (if possible)
        try:
            self.ll_history.append(float(loglik.numpy()))
        except Exception:
            try:
                self.ll_history.append(float(loglik))
            except Exception:
                self.ll_history.append(None)

        # RMSE against provided true state (if given)
        if true_state is not None:
            try:
                true_t = tf.convert_to_tensor(true_state, dtype=self.dtype)
                diff = tf.reshape(self.m - true_t, [-1])
                rmse = tf.sqrt(tf.reduce_mean(tf.square(diff)))
                try:
                    self.rmse_history.append(float(rmse.numpy()))
                except Exception:
                    self.rmse_history.append(float(rmse))
            except Exception:
                self.rmse_history.append(None)

        return self.m, self.P, loglik

    def _jacobian(self, func: Callable[[tf.Tensor], tf.Tensor], x: tf.Tensor) -> tf.Tensor:
        """Compute Jacobian of func at x using tf.GradientTape.

        Returns tensor of shape (out_dim, in_dim).
        """
        x = tf.convert_to_tensor(x, dtype=self.dtype)
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = tf.convert_to_tensor(func(x), dtype=self.dtype)
        J = tape.jacobian(y, x)
        # Ensure 2D (out_dim, in_dim)
        J = tf.reshape(J, (tf.shape(y)[0], tf.shape(x)[0]))
        return J

    def filter(self, observations, true_states=None) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Run the Extended Kalman Filter on observations.

        Parameters
        ----------
        observations : array-like or tf.Tensor of shape (T, obs_dim) or (T,) for 1D

        Returns
        -------
        filtered_means : tf.Tensor, shape (T, state_dim)
        filtered_covs : tf.Tensor, shape (T, state_dim, state_dim)
        loglik : tf.Tensor scalar
        """
        obs_arr = tf.convert_to_tensor(observations, dtype=self.dtype)
        # normalize shape to (T, obs_dim)
        if obs_arr.shape.rank == 1:
            obs_arr = tf.reshape(obs_arr, (-1, 1))

        T = tf.shape(obs_arr)[0]

        # Optional true states for RMSE computation. Accepts array-like of shape (T, state_dim) or (T,)
        true_states_tensor = None
        if true_states is not None:
            try:
                true_states_tensor = tf.convert_to_tensor(true_states, dtype=self.dtype)
                if true_states_tensor.shape.rank == 1:
                    # reshape to (T, state_dim=1)
                    true_states_tensor = tf.reshape(true_states_tensor, (-1, 1))
            except Exception:
                true_states_tensor = None

        m = tf.identity(self.m0)
        P = tf.identity(self.P0)

        filtered_means = tf.TensorArray(self.dtype, size=T)
        filtered_covs = tf.TensorArray(self.dtype, size=T)

        total_loglik = tf.constant(0.0, dtype=self.dtype)
        I = tf.eye(self.state_dim, dtype=self.dtype)

        for t in tf.range(T):
            y = obs_arr[t]

            # Predict
            m_pred = tf.convert_to_tensor(self.f(m), dtype=self.dtype)
            F = self._jacobian(self.f, m)
            P_pred = F @ P @ tf.transpose(F) + self.Q

            # Innovation
            y_pred = tf.convert_to_tensor(self.h(m_pred), dtype=self.dtype)
            H = self._jacobian(self.h, m_pred)
            S = H @ P_pred @ tf.transpose(H) + self.R

            # Kalman gain via Cholesky solve for numerical stability
            chol_S = tf.linalg.cholesky(S)
            K = tf.linalg.matmul(P_pred, tf.transpose(H))
            K = tf.linalg.cholesky_solve(chol_S, tf.transpose(K))
            K = tf.transpose(K)

            innovation = y - y_pred

            # Update
            m = m_pred + tf.linalg.matvec(K, innovation)
            KH = K @ H
            P = (I - KH) @ P_pred @ tf.transpose(I - KH) + KH @ self.R @ tf.transpose(KH)

            # Log-likelihood
            mvn = tfd.MultivariateNormalFullCovariance(loc=y_pred, covariance_matrix=S)
            loglik = mvn.log_prob(y)
            total_loglik += loglik

            # Record per-step diagnostics
            try:
                self.ll_history.append(float(loglik.numpy()))
            except Exception:
                try:
                    self.ll_history.append(float(loglik))
                except Exception:
                    self.ll_history.append(None)

            # RMSE if true states provided
            if true_states_tensor is not None:
                try:
                    true_t = true_states_tensor[t]
                    diff = tf.reshape(m - tf.convert_to_tensor(true_t, dtype=self.dtype), [-1])
                    rmse = tf.sqrt(tf.reduce_mean(tf.square(diff)))
                    try:
                        self.rmse_history.append(float(rmse.numpy()))
                    except Exception:
                        self.rmse_history.append(float(rmse))
                except Exception:
                    self.rmse_history.append(None)

            filtered_means = filtered_means.write(t, m)
            filtered_covs = filtered_covs.write(t, P)

        filtered_means = filtered_means.stack()
        filtered_covs = filtered_covs.stack()

        return filtered_means, filtered_covs, total_loglik


def _simulate_simple_nonlinear(T=50, seed: int = 1):
    """Simulate a 1D mild nonlinear system for tests.

    This helper returns NumPy arrays for true states and observations,
    plus scalar Q and R values.
    """
    rng = np.random.default_rng(seed)
    x = np.zeros(T + 1)
    y = np.zeros(T)
    Q = 0.01
    R = 0.1
    x[0] = 0.5
    for t in range(1, T + 1):
        process = rng.normal(scale=np.sqrt(Q))
        x[t] = x[t - 1] + 0.05 * (x[t - 1] ** 2) + process
    for t in range(T):
        obs = x[t + 1] + 0.1 * (x[t + 1] ** 2) + rng.normal(scale=np.sqrt(R))
        y[t] = obs
    return x[1:], y, Q, R


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    true_x, observations, Q, R = _simulate_simple_nonlinear(T=100)

    def f(x):
        return x + 0.05 * tf.square(x)

    def h(x):
        return x + 0.1 * tf.square(x)

    ekf = ExtendedKalmanFilter(
        f=f,
        h=h,
        Q=tf.constant([[Q]], dtype=tf.float64),
        R=tf.constant([[R]], dtype=tf.float64),
        initial_mean=tf.constant([0.0], dtype=tf.float64),
        initial_cov=tf.constant([[1.0]], dtype=tf.float64),
    )

    means, covs, ll = ekf.filter(observations)
    print("Log-likelihood:", float(ll))

    plt.plot(true_x, label="true")
    plt.plot(means.numpy().squeeze(), label="ekf mean")
    plt.legend()
    plt.show()

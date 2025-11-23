"""Unscented Kalman filter implementation using TensorFlow and TensorFlow Probability.

This module provides an Unscented Kalman Filter (UKF) implementation that uses
sigma points to handle nonlinear state transitions and observations. It uses
TensorFlow for tensor operations and TensorFlow Probability for distributions.

Class
------
- UnscentedKalmanFilter: Construct with system functions and run filter() on observations.

Example
-------
>>> import numpy as np
>>> import tensorflow as tf
>>> from src.models.UnscentedKalmanFilter import UnscentedKalmanFilter
>>> # See demo at bottom of file or run python -m src.models.UnscentedKalmanFilter
"""
from __future__ import annotations

from typing import Callable, Optional, Tuple, List

import tensorflow as tf
import tensorflow_probability as tfp

tfb = tfp.bijectors
tfd = tfp.distributions


class UnscentedKalmanFilter:
    """Unscented Kalman Filter using TensorFlow operations.

    This implements the discrete-time UKF for the model:
        x_t = f(x_{t-1}) + q_t,   q_t ~ N(0, Q)
        y_t = h(x_t) + r_t,       r_t ~ N(0, R)

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
    alpha : float, optional
        UKF parameter controlling spread of sigma points (default: 0.001)
    beta : float, optional
        UKF parameter for prior knowledge of state distribution (2 for Gaussian)
    kappa : float, optional
        UKF parameter (usually 3-n for n dimensions), default: 0
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
        alpha: float = 0.001,
        beta: float = 2.0,
        kappa: Optional[float] = None,
        dtype: tf.DType = tf.float64,
        verbose_or_not: bool = False,
    ) -> None:
        # Save functions and dtype
        self.f = f
        self.h = h
        self.dtype = dtype

        # Convert inputs to tensors
        self.Q = tf.convert_to_tensor(Q, dtype=self.dtype)
        self.R = tf.convert_to_tensor(R, dtype=self.dtype)
        self.m0 = tf.convert_to_tensor(initial_mean, dtype=self.dtype)
        self.P0 = tf.convert_to_tensor(initial_cov, dtype=self.dtype)

        # Get dimensions
        self.state_dim = int(self.P0.shape[0])
        self.obs_dim = int(self.R.shape[0])

        # Validate dimensions
        assert tuple(self.Q.shape) == (self.state_dim, self.state_dim)
        assert tuple(self.R.shape) == (self.obs_dim, self.obs_dim)

        # Initialize UKF parameters with proper type conversion
        n = float(self.state_dim)
        self.alpha = float(alpha)
        self.beta = float(beta)
        # If kappa is not provided, use the common default 3 - n
        self.kappa = float(kappa) if (kappa is not None) else float(3.0 - n)

        # Calculate lambda parameter (as a Python float)
        self.lambda_ = (self.alpha ** 2) * (n + self.kappa) - n

        # Calculate weights
        # Obtained as tf.Tensor attributes: self.wm, self.wc
        
        self._calculate_weights()
        # initialize running state for step() convenience (keeps filter stateful)
        self.m = tf.identity(self.m0)
        self.P = tf.identity(self.P0)
        self.verbose = bool(verbose_or_not)
        # Evaluation histories
        self.rmse_history = []
        self.ll_history = []

    def reset(self) -> None:
        """Reset the internal filter state to the initial mean/covariance."""
        self.m = tf.identity(self.m0)
        self.P = tf.identity(self.P0)

    def step(self, observation, true_state=None) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Perform a single predict-update step (stateful) and return posterior.

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
        # normalize shape for scalar observations
        if y.shape.rank == 0:
            y = tf.reshape(y, (1,))

        # Generate sigma points from current state
        chi = self._generate_sigma_points(self.m, self.P)

        if self.verbose:
            print("UnscentedKalmanFilter: step() generating sigma points and performing predict-update")

        # Predict step: propagate sigma points through dynamics
        chi_pred = tf.map_fn(self.f, chi)
        m_pred = tf.reduce_sum(self.wm[:, None] * chi_pred, axis=0)

        # Covariance prediction
        diffs = chi_pred - m_pred
        wc_reshaped = tf.reshape(self.wc, [-1, 1])
        P_pred = tf.transpose(diffs) @ (wc_reshaped * diffs)
        P_pred = tf.cast(P_pred, dtype=self.dtype) + self.Q

        # Observation prediction
        chi_y = self._generate_sigma_points(m_pred, P_pred)
        gamma = tf.map_fn(self.h, chi_y)
        y_pred = tf.reduce_sum(self.wm[:, None] * gamma, axis=0)

        diffs_x = chi_y - m_pred
        diffs_y = gamma - y_pred
        wc_reshaped = tf.reshape(self.wc, [-1, 1])

        S = tf.transpose(diffs_y) @ (wc_reshaped * diffs_y)
        Pxy = tf.transpose(diffs_x) @ (wc_reshaped * diffs_y)
        S = tf.cast(S, dtype=self.dtype) + self.R

        # Kalman gain and update
        K = Pxy @ tf.linalg.inv(S)
        innovation = y - y_pred
        self.m = m_pred + tf.linalg.matvec(K, innovation)

        # Joseph-form covariance update for numerical stability.
        # Estimate an effective linearized observation matrix H such that
        # Pxy = P_pred @ H^T  => H^T = P_pred^{-1} @ Pxy
        # Then KH = K @ H and Joseph form: (I - KH) P_pred (I - KH)^T + K R K^T
        P_pred_inv = tf.linalg.inv(P_pred)
        H_est = tf.transpose(P_pred_inv @ Pxy)  # shape (obs_dim, state_dim)
        KH = K @ H_est
        I = tf.eye(self.state_dim, dtype=self.dtype)
        self.P = (I - KH) @ P_pred @ tf.transpose(I - KH) + K @ self.R @ tf.transpose(K)
        # Enforce symmetry to reduce numerical asymmetry
        self.P = 0.5 * (self.P + tf.transpose(self.P))

        # Log-likelihood for this observation
        mvn = tfd.MultivariateNormalFullCovariance(loc=y_pred, covariance_matrix=S)
        loglik = mvn.log_prob(y)

        # Record per-step diagnostics
        try:
            self.ll_history.append(float(loglik.numpy()))
        except Exception:
            try:
                self.ll_history.append(float(loglik))
            except Exception:
                self.ll_history.append(None)

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

    def _calculate_weights(self) -> None:
        """Calculate UKF weights for mean and covariance."""
        n = float(self.state_dim)
        lambda_ = float(self.lambda_)

        # Tensor versions
        n_t = tf.constant(n, dtype=self.dtype)
        lambda_t = tf.constant(lambda_, dtype=self.dtype)

        # Weight for the zeroth sigma point
        wm0 = lambda_t / (n_t + lambda_t)
        # Weight for the other sigma points
        w_other = tf.constant(1.0 / (2.0 * (n + lambda_)), dtype=self.dtype)

        # Build weight vectors (length = 2n + 1)
        m = int(2 * int(n) + 1)
        wm_rest = tf.fill([m - 1], w_other)
        self.wm = tf.concat([tf.reshape(wm0, [1]), wm_rest], axis=0)

        # Covariance weights: adjust zeroth element
        wc0 = wm0 + tf.constant(1.0 - self.alpha ** 2 + self.beta, dtype=self.dtype)
        wc_rest = wm_rest
        self.wc = tf.concat([tf.reshape(wc0, [1]), wc_rest], axis=0)

    def _generate_sigma_points(self, mean: tf.Tensor, cov: tf.Tensor) -> tf.Tensor:
        """Generate sigma points using the scaled unscented transform.
        
        Parameters
        ----------
        mean : tf.Tensor of shape (state_dim,)
            Current state mean
        cov : tf.Tensor of shape (state_dim, state_dim)
            Current state covariance
            
        Returns
        -------
        sigma_points : tf.Tensor of shape (2*state_dim + 1, state_dim)
            Matrix of sigma points
        """
        n = int(self.state_dim)
        lambda_ = float(self.lambda_)

        # Cholesky of covariance (n x n)
        U = tf.linalg.cholesky(cov)  # lower-triangular
        scaling = tf.sqrt(tf.constant(n + lambda_, dtype=self.dtype))
        scaled_U = scaling * U

        # Build sigma points: center, then mean +/- columns of scaled_U
        sigmas: List[tf.Tensor] = []
        sigmas.append(mean)
        for i in range(n):
            col = scaled_U[:, i]
            sigmas.append(mean + col)
        for i in range(n):
            col = scaled_U[:, i]
            sigmas.append(mean - col)

        return tf.stack(sigmas, axis=0)

    def filter(self, observations, true_states=None) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Run the Unscented Kalman Filter on observations.

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
        m = tf.identity(self.m0)
        P = tf.identity(self.P0)
        
        filtered_means = tf.TensorArray(dtype=self.dtype, size=T)
        filtered_covs = tf.TensorArray(dtype=self.dtype, size=T)
        total_loglik = tf.constant(0.0, dtype=self.dtype)
        
        # Optional true states tensor for RMSE computation
        true_states_tensor = None
        if true_states is not None:
            try:
                true_states_tensor = tf.convert_to_tensor(true_states, dtype=self.dtype)
                if true_states_tensor.shape.rank == 1:
                    true_states_tensor = tf.reshape(true_states_tensor, (-1, 1))
            except Exception:
                true_states_tensor = None

        for t in tf.range(T):
            # Get observation
            y = obs_arr[t]
            
            # Generate sigma points
            chi = self._generate_sigma_points(m, P)

            # Predict step: propagate sigma points through dynamics
            chi_pred = tf.map_fn(self.f, chi)
            # mean prediction
            m_pred = tf.reduce_sum(self.wm[:, None] * chi_pred, axis=0)

            # Covariance prediction: vectorized form
            diffs = chi_pred - m_pred  # shape (2n+1, n)
            wc_reshaped = tf.reshape(self.wc, [-1, 1])  # (2n+1, 1)
            P_pred = tf.transpose(diffs) @ (wc_reshaped * diffs)
            P_pred = tf.cast(P_pred, dtype=self.dtype) + self.Q
            
            # Generate sigma points for observation prediction
            chi = self._generate_sigma_points(m_pred, P_pred)
            gamma = tf.map_fn(self.h, chi)

            # Predict observation
            y_pred = tf.reduce_sum(self.wm[:, None] * gamma, axis=0)

            # Innovation covariance and cross-covariance (vectorized)
            diffs_x = chi - m_pred  # (2n+1, n)
            diffs_y = gamma - y_pred  # (2n+1, obs_dim)
            wc_reshaped = tf.reshape(self.wc, [-1, 1])

            S = tf.transpose(diffs_y) @ (wc_reshaped * diffs_y)
            Pxy = tf.transpose(diffs_x) @ (wc_reshaped * diffs_y)
            S = tf.cast(S, dtype=self.dtype) + self.R
            
            # Kalman gain
            K = Pxy @ tf.linalg.inv(S)

            # Update
            innovation = y - y_pred
            m = m_pred + tf.linalg.matvec(K, innovation)

            # Joseph-form covariance update (estimate H via sigma-point covariances)
            P_pred_inv = tf.linalg.inv(P_pred)
            H_est = tf.transpose(P_pred_inv @ Pxy)
            KH = K @ H_est
            I = tf.eye(self.state_dim, dtype=self.dtype)
            P = (I - KH) @ P_pred @ tf.transpose(I - KH) + K @ self.R @ tf.transpose(K)
            P = 0.5 * (P + tf.transpose(P))
            
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
    


def _simulate_van_der_pol(T=50, seed: int = 1):
    """Simulate the Van der Pol oscillator for testing.
    
    dx1/dt = x2
    dx2/dt = mu(1-x1^2)x2 - x1
    
    We'll use Euler integration and add noise.
    """
    import numpy as np
    
    dt = 0.1  # integration step
    mu = 1.0  # oscillator parameter
    
    rng = np.random.default_rng(seed)
    
    # Initialize
    x = np.zeros((T+1, 2))  # [position, velocity]
    y = np.zeros((T, 2))    # observe both states with noise
    x[0] = [1.0, 0.0]      # start at x=1, v=0
    
    # System and observation noise
    Q = np.array([[0.01, 0.0], [0.0, 0.01]])  # state noise
    R = np.array([[0.1, 0.0], [0.0, 0.1]])    # observation noise
    
    # Simulate
    for t in range(T):
        # Euler integration with noise
        dx1 = x[t,1]
        dx2 = mu*(1 - x[t,0]**2)*x[t,1] - x[t,0]
        
        x[t+1,0] = x[t,0] + dx1*dt + rng.multivariate_normal([0,0], Q)[0]*np.sqrt(dt)
        x[t+1,1] = x[t,1] + dx2*dt + rng.multivariate_normal([0,0], Q)[1]*np.sqrt(dt)
        
        # Noisy observation
        y[t] = x[t+1] + rng.multivariate_normal([0,0], R)
    
    return x[1:], y, Q, R


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Simulate Van der Pol oscillator
    true_x, observations, Q, R = _simulate_van_der_pol(T=100)
    
    # Define system functions for UKF
    def f(x: tf.Tensor) -> tf.Tensor:
        """Van der Pol oscillator dynamics"""
        dt = 0.1
        mu = 1.0
        dx1 = x[1]
        dx2 = mu*(1 - x[0]**2)*x[1] - x[0]
        return x + tf.stack([dx1, dx2])*dt
    
    def h(x: tf.Tensor) -> tf.Tensor:
        """Identity observation"""
        return x
    
    # Create and run UKF
    ukf = UnscentedKalmanFilter(
        f=f,
        h=h,
        Q=tf.constant(Q, dtype=tf.float64),
        R=tf.constant(R, dtype=tf.float64),
        initial_mean=tf.constant([1.0, 0.0], dtype=tf.float64),
        initial_cov=tf.constant([[1.0, 0.0], [0.0, 1.0]], dtype=tf.float64),
    )
    
    means, covs, ll = ukf.filter(observations)
    print("Log-likelihood:", float(ll))
    
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.plot(true_x[:,0], true_x[:,1], 'k-', label='True')
    plt.plot(means.numpy()[:,0], means.numpy()[:,1], 'r--', label='UKF')
    plt.plot(observations[:,0], observations[:,1], 'k.', alpha=0.3, label='Obs')
    plt.xlabel('Position')
    plt.ylabel('Velocity')
    plt.title('Van der Pol Phase Space')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(122)
    t = np.arange(len(true_x))
    plt.plot(t, true_x[:,0], 'k-', label='True position')
    plt.plot(t, means.numpy()[:,0], 'r--', label='UKF position')
    plt.plot(t, observations[:,0], 'k.', alpha=0.3, label='Observations')
    plt.xlabel('Time step')
    plt.ylabel('Position')
    plt.title('Position vs Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()